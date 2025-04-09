import re
import os
import sys
import traceback
sys.path.append(os.getcwd())

import asyncio
from tenacity import (
    retry,
    wait_random_exponential,
)  
import json
from openai import OpenAIError, RateLimitError, APIConnectionError
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils.config import settings
from app.utils.helpers.openai_token_counter import count_tokens_text, count_tokens_messages, get_available_tokens, are_tokens_available_for_both_conversations
from app.utils.helpers.prompt_helper import create_question_extraction_conversation_messages, create_answer_generation_conversation_messages, create_text_cleaning_conversation_messages
from app.utils.logger import setup_logger
logger = setup_logger()
# Ensure we do not run too many concurent requests

class QAGenerator:
    def __init__(self):
        self.chunk_size = 2000
        self.chunk_overlap = 200
        self.model_rate_limits = 2000
        self.max_concurent_request = int(self.model_rate_limits * 0.75)
        self.throttler = asyncio.Semaphore(self.max_concurent_request)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.service_name = "qa_generator_service"


    def extract_questions_from_output(self, output):
        """
        Takes a numbered list of questions as a string and returns them as a list of strings.
        The input might have prefixes/suffixes that are not questions or incomplete questions.

        Args:
            output (str): A string containing a numbered list of questions.

        Returns:
            list of str: A list of extracted questions as strings.
        """
        try:
            # Define a regex pattern to match questions (lines starting with a number followed by a dot and a space)
            question_pattern = re.compile(r"^\s*\d+\.\s*(.+)$", re.MULTILINE)

            # Find all the questions matching the pattern in the input text
            questions = question_pattern.findall(output)

            # Check if the last question is incomplete (does not end with punctuation or a parenthesis)
            if (len(questions) > 0) and (not re.search(r"[.!?)]$", questions[-1].strip())):
                print(f"WARNING: Popping incomplete question: '{questions[-1]}'")
                questions.pop()
            return questions
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return []


    # Text Chunking Function
    def chunk_text(self, text):
        try:
            return self.text_splitter.split_text(text)
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return [text]


    # Run Model Function
    @retry(wait=wait_random_exponential(min=15, max=40))
    async def run_model(self, messages):
        """
        Asynchronously runs the chat model while respecting token limits.

        Args:
            messages (list): List of input messages for the model.

        Returns:
            str: Model-generated output text.
        """
        try:
            num_tokens_in_messages = count_tokens_messages(messages)
            num_tokens_available = get_available_tokens(num_tokens_in_messages)

            # Check token availability before calling the model
            if num_tokens_available <= 0:
                logger.exception("ERROR: Insufficient tokens available for this request.", extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
                return 'ERROR'

            model = ChatOpenAI(model=settings.BASE_MODEL, temperature=0.0, max_tokens=num_tokens_available)

            try:
                async with self.throttler:
                    output = await model._agenerate(messages)
            except (RateLimitError, APIConnectionError) as error1:
                logger.exception(error1, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
                raise error1
            except Exception as error2:
                logger.exception(error2, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
                return 'ERROR'

            return output.generations[0].text.strip()
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return 'ERROR'
    
    
    # Text Cleaning Function
    async def clean_text(self, text):
        try:
            messages = create_text_cleaning_conversation_messages(text)
            return await self.run_model(messages)
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return text

    # Flatten Nested Lists Function
    def flatten_nested_lists(self, nested_lists):
        """
        Takes a list of lists as input and returns a flattened list containing all elements.
        
        Args:
            nested_lists (list of lists): A list containing one or more sublists.

        Returns:
            list: A flattened list containing all elements from the input nested lists.
        """
        flattened_list = []
        try:
            # Iterate through the nested lists and add each element to the flattened_list
            for sublist in nested_lists:
                flattened_list.extend(sublist)
            return flattened_list
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return flattened_list


    # Extract Questions from Text Function
    async def extract_questions_from_text(self, text):
        """
        Asynchronously extracts questions from text while handling token limits.

        Args:
            file_path (str): Markdown file path.
            text (str): Text content of the file.

        Returns:
            list of tuples: [(file_path, text, question), ...]
        """
        try:
            text = await self.clean_text(text)
            text = text.strip()
            num_tokens_text = count_tokens_text(text)

            if not are_tokens_available_for_both_conversations(num_tokens_text):
                logger.warning("WARNING: Splitting the chunk into smaller chunks.", extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
                tasks = []
                self.chunk_size = 1500
                self.chunk_overlap = 100
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", " ", ""]
                )
                for sub_text in self.chunk_text(text):
                    tasks.append(self.extract_questions_from_text(sub_text))
                tasks_outputs = await asyncio.gather(*tasks)
                self.chunk_size = 2000
                self.chunk_overlap = 200
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=["\n\n", "\n", " ", ""]
                )
                return self.flatten_nested_lists(tasks_outputs)
            # Run the model if tokens are sufficient
            messages = create_question_extraction_conversation_messages(text)
            output = await self.run_model(messages)
            questions = self.extract_questions_from_output(output)
            return [(text, question.strip()) for question in questions]
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return []


    # Generate Answer Function
    async def generate_answer(self, question, context):
        try:
            messages = create_answer_generation_conversation_messages(question, context)
            return await self.run_model(messages)
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return 'ERROR'

    # Full Pipeline
    async def generate_question_and_answer(self, text):
        try:
            chunks = self.chunk_text(text)
            async def process_chunk(chunk):
                try:
                    questions = await self.extract_questions_from_text(chunk)
                    # print(f"Extracted {len(questions)} questions from the chunk.")
                    with ThreadPoolExecutor() as executor:
                        answers = await asyncio.gather(
                            # *[loop.run_in_executor(executor, generate_answer, q[1], chunk) for q in questions]
                            *[self.generate_answer(q[1], chunk) for q in questions]
                        )
                        # print(f"Generated {len(answers)} answers for the chunk.")
                    return [(question[1], answer) for question, answer in zip(questions, answers)]
                except Exception as error1:
                    logger.exception(error1, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
                    return []
            qa_pairs = await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])
            qa_pairs = self.flatten_nested_lists(qa_pairs)
            qa_pairs = [{"question": question, "answer": answer} for question, answer in qa_pairs]
            return qa_pairs
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return []

# Example Usage
async def process_text():
    try:
        text = """
                                            A N T O I N E   S A L L I S
            that the folks in the school’s administration office have not answered. These
            are questions that are not explained to us in school. Why is this? There has
            to be a reason. It cannot be a coincidence.
                Banks  do not like credit repair companies. How do we know? One
            reason is that if a credit repair company opens up an account, it gets shut
            down most of the time. Another reason we know that banks do not like
            credit repair companies is that a bank will not give a credit repair company
            a merchant account. They see it as a conflict of interest. So, in essence, a
            bank feels competition with a company helping clients fix and repair their
            credit profile. Go figure!
                The United States of America is the most powerful coun       try in the
            world. One of the things America prides itself on is the educational system.
            If this is so, how are we missing such a huge component of our lives by not
            teaching credit in school? Have you ever wondered that? Credit is one of
            the biggest factors in the U.S. In fact, statistics show credit is 75% of every
            buying decision that you make. So with that being said, the fact that they
            don’t teach us this in school, the fact that they don’t pound this into our
            brains, what does that mean? Is it an accident? Did they accidentally forget
            to tell us about the most important thing in our lives? The truth is no. It’s
            not an accident.
                The truth is that credit is meant to naturally and organically be utilized
            so that the bureaus can dictate what your spending pa tterns are and what
            your payback patterns are. It’s risk assessment that is assessed. Do you
            naturally pay your bills on time? Do you naturally open up different types
            of accounts? They do not want you to strategically place your credit and
            build your credit to a certain aspect to get approved for things in life that
            will give you low interest rates, high credit limits, and high revolving
                                                - 2 -
            ---
                        T H E   G R E A T   A M E R I C A N   C R E D I T   S E C R E T
            amounts. They want that to be a natural thing. They do not want you to
            know how to formulate your credit profile on your  own. This is why they
            do not want credit taught in schools. This is why it’s a secret.
                The average consumer does not truly realize or understand the depth,
            the gravity, and the power of credit. If they did, then with the same energy,
            the same determination, and the same passion that students use to earn a
            degree, and the same passion that compels the average human being to
            spend hundreds of thousands of dollars on their education, they would
            utilize that same energy for building the most powerful credit sc  ore ever
            (And it wouldn’t even take 10% of that effort). There’s only a very, very
            small sliver of Americans chasing credit like their lives depend on it because
            they understand how powerful credit truly is. These individuals who do
            understand are a small    percentage of the country’s population. These
            individuals truly know and understand The Great American Credit Secret.
                Here’s an example: Let’s say I’m going to school with one of my best
            friends, and then his parents add him to their credit card as an Authorized
            User. His parents’ credit card will show up on his credit report. So if they’ve
            had a good payment history and had on     -time payments and a big credit
            limit, that will show up on my friend’s credit report. So then suppose I say
            to my friend, “Hey, let’s go get a cell phone,” and we go to the Verizon store.
            We’re the same age. We’ve been buddies for years. We both want the
            newest iPhone. The Verizon salesperson tells my friend, “Well, sir, you are
            approved for the iPhone and you only have to pay the tax es on it, which is
            only $130.”
                I see my friend with his new cell phone, and I say, “Wow, this is great,
            this is great! Yes! Sign me up as well.”
                                            - 3 -
            ---
                                        A N T O I N E   S A L L I S
                However, when they run my credit to sign me up, they see there’s
            nothing on the credit file. And since there’s nothing on a credit file, Verizon
            says, “Yes, you are approved, but… you just have to put down 50% of the
            phone’s value.”
                And I say, “Oh, what does that mean?”
                “Well, the phone is $1,300, so you have to pay $675.”
                I look at my friend, but now I’m 100% conf    used. In my mind, I’m
            thinking, What just happened here? Why do I have to pay more than him?
            Why didn’t I qualify with the minimum down like my friend?
                This is an everyday scenario. It comes from ignorance of not truly
            knowing and understanding the power o f credit. The sad part about this
            illustration is that it is only a small example of why having good credit is
            important. On a larger scale, people with bad credit pay up to millions of
            dollars more over a lifetime than someone with good credit pays. It’s kind
            of ironic if you truly think about the system. Some may say that it has a
            glitch.
                If someone has good credit, the typical person would think that they
            have the money to afford their bills, so why should someone who does not
            have money to pay their bills and has bad credit, why should they pay more?
            Well, ironically, that essentially is exactly what is happening. Many would
            argue that if someone has bad credit, it’s a bigger risk for the lender, so they
            have to charge higher interest—I beg to differ. Th ose higher interest rates
            keep many banks in business, and if those with bad credit are being charged
            higher interest rates, it essentially makes them have to dig a deeper hole in
            the credit world.
                                    - 4 -
        """
        qa_generator = QAGenerator()
        result = await qa_generator.generate_question_and_answer(text)
        with open(os.path.join(settings.LOCAL_UPLOAD_LOCATION,'qa_pair_result.json'), 'w') as f:
            json.dump(result, f, indent=4)
        # print(result)
    except Exception as e:
        print(f"process_text:: ERROR: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(process_text())
