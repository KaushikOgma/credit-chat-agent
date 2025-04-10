import asyncio
import json
import traceback
from typing import List, Union
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from tqdm import tqdm
from app.services.pinecone_vectorizer import OpenAIEmbedding, VectorizerEngine
from app.utils.helpers.prompt_helper import chat_system_content_message
from app.utils.helpers.request_helper import start_chat
from app.utils.config import settings
from app.services.chat_service import ChatService
import openai
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from app.utils.logger import setup_logger
logger = setup_logger()


class QAEvaluator:
    """
    A class to evaluate the quality of the AI-generated responses.
    """
    def __init__(self):
        # Initialize the OpenAI Embedding
        self.encoder = OpenAIEmbedding(model_name=settings.EMBEDDING_MODEL_NAME)
        # Initialize the Vectorizer Engine
        self.temperature = 0.7
        self.max_tokens = 500
        self.openai = openai
        self.openai.api_key = settings.OPENAI_API_KEY
        self.vectorizer = VectorizerEngine(
            encoder=self.encoder,
            vector_db_name=settings.VECTOR_DB_NAME,
            batch_size=10,
            dimension=settings.VECTOR_DIMENSION,
            namespace="questions"
        )
        self.similarity_threshold = 0.8
        self.system_prompt = chat_system_content_message()
        self.chat_service = ChatService()
        self.service_name = "model_evaluator_service"

    async def get_precision_recall_f1(self, generated_sim_scores: List[float], true_sim_scores: List[float]) -> Union[dict, None]:
        """
        
        Compute the precision, recall, and F1 score based on the similarity scores

        Args:
            generated_sim_scores (List[float]): List of generated similarity scores
            true_sim_scores (List[float]): List of true similarity scores            

        Returns:
            dict: Average score, standard deviation, coefficient of variation, precision, recall, and F1 score
        """
        score = {}
        try:
            # Compute average score and standard deviation
            average_score = np.mean(generated_sim_scores)
            std_dev = np.std(generated_sim_scores)
            cv = (std_dev / average_score) * 100

            # Compute precision, recall, and F1 score
            # y_true = [1] * len(sim_scores)  # Assuming all ground truths are 'correct'
            y_true = [1 if score >= self.similarity_threshold else 0 for score in true_sim_scores]
            y_pred = [1 if score >= self.similarity_threshold else 0 for score in generated_sim_scores]
            precision = precision_score(y_true, y_pred, zero_division=0) * 100
            recall = recall_score(y_true, y_pred, zero_division=0) * 100
            f1 = f1_score(y_true, y_pred, zero_division=0) * 100
            accuracy = accuracy_score(y_true, y_pred) * 100

            score = {
                "average_score": round(average_score, 3),
                "std_dev": round(std_dev, 3),
                "cv": round(cv, 3),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "accuracy": round(accuracy, 3)
            }
            print("--------------------------------------------------------------")
            print("                      Evaluation Results                      ")
            print("--------------------------------------------------------------")
            print(f"Average Score (Mean): {average_score:.2f}%")
            print(f"Standard Deviasion (std): {std_dev:.2f}")
            print(f"Coefficient of Variation (CV): {cv:.2f}")
            print(f"precision: {precision:.2f}%")
            print(f"recall: {recall:.2f}%")
            print(f"f1: {f1:.2f}%")
            print(f"accuracy: {accuracy:.2f}%")
            print("--------------------------------------------------------------")
            return score
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return score

    async def sync_vector_db(self, qa_pairs: List[dict]) -> bool: 
        """
        Sync the vector DB with the latest data.
        """
        try:
            # Load the vector store
            self.vectorizer.load_vectorstore()
            # Sync the vector DB with the latest QA pairs
            await self.vectorizer.create_vectorstore(qa_pairs, "question_id", "question")
            await asyncio.sleep(3)
            return True
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return False
        finally:
            # Unload the vector store
            self.vectorizer.unload_vectorstore()
        

    async def delete_eval_data_from_vector_db(self, id_list: List[str]) -> bool: 
        """
        Sync the vector DB with the latest data.
        """
        try:
            # Load the vector store
            self.vectorizer.load_vectorstore()
            # Sync the vector DB with the latest QA pairs
            await self.vectorizer.delete_items(id_list)
            return True
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return False
        finally:
            # Unload the vector store
            self.vectorizer.unload_vectorstore()
        

    async def evaluate_single_qa(self, question: str, model_id: str) -> float:
        """
        Evaluate a single QA by checking if the AI response is present in the vector DB.

        Args:
            question (str): Input question

        Returns:
            dict: Similarity score and matched response from the vector DB.
        """
        try:
            # Load the vector store
            self.vectorizer.load_vectorstore()
            # Get the AI response for the question
            if settings.TEST_USER_ID is not None and len(settings.TEST_USER_ID) > 0:
                resp = await start_chat(question)
                print("----------------------------------------")
                print(json.dumps(resp, indent=2))
                print("----------------------------------------")
                if resp["success"]:
                    ai_response = resp["data"]["response"]
                else:
                    ai_response = None
            else:
                ai_response = await self.chat_service.get_response(question, model_id)
            if ai_response:
                # Get the similarity score between the question actual answer and the AI response
                generated_similarity_score, true_similarity_score = await self.vectorizer.get_qa_similarity_score(
                    question=question,
                    answer=ai_response
                )
                return ai_response, round(generated_similarity_score, 3), round(true_similarity_score, 3)
            else:
                return None, 0.0, 0.0
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return None, 0.0, 0.0
        finally:
            # Unload the vector store
            self.vectorizer.unload_vectorstore()
    

    async def evaluate_qa_pairs(self, qa_pairs: List[dict], model_id: str) -> Union[dict, None]:
        """
        Evaluate a list of QA pairs by checking if the AI response is present in the vector DB.

        Args:
            qa_pairs (List[dict]): List of QA pairs

        Returns:
            dict: Similarity score and matched response from the vector DB.
        """
        try:
            # print("qa_pairs:: ",json.dumps(qa_pairs, indent=4))
            generated_score_list = []
            generated_answers = {}
            true_score_list = []
            agg_scores = None
            is_synced = await self.sync_vector_db(qa_pairs)
            if is_synced:
                for qa_pair in tqdm(qa_pairs, desc="Evaluating QA pairs"):
                    question = qa_pair['question'].strip()
                    generated_answer, generated_score, true_score = await self.evaluate_single_qa(question, model_id)
                    generated_score_list.append(generated_score)
                    true_score_list.append(true_score)
                    generated_answers[question] = {"answer": generated_answer, "similarity_score": generated_score}
                agg_scores = await self.get_precision_recall_f1(generated_score_list,true_score_list)
                agg_scores["eval_sample_count"] = len(qa_pairs)
            return {
                "generated_answers": generated_answers,
                "agg_scores": agg_scores
            }
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return None
        




async def start_evaluation():
    # Initialize the fine-tuner
    qa_evaluator = QAEvaluator()
    model_id = "ft:gpt-4o-2024-08-06:the-great-american-credit-secret-llc::BABaftly"
    # Sample Q&A data for testing
    qa_pairs = [
        {
            "question_id": "f3b82a10-26d2-4c8f-b4b8-b71f3d3a80a4",
            "question": "What exactly is credit?",
            "answer": "It's the system of borrowing money with the agreement to pay it back later, often with interest."
        },
        {
            "question_id": "7e1e5a5a-6e6d-4a67-9d38-2d5f10b3a1b5",
            "question": "Why should I care about credit?",
            "answer": "Good credit can help you get loans with better interest rates and is essential for big purchases like a car or home."
        },
        {
            "question_id": "c8b3b0cb-d28b-49a9-95d5-bd6a7c0e4b2d",
            "question": "What’s a credit score?",
            "answer": "A number that lenders use to determine how risky it is to lend you money, based on your credit history."
        },
        {
            "question_id": "9d7a5cb5-d6b2-4a1c-8c58-f845db22cf84",
            "question": "How can I improve my credit score?",
            "answer": "Make payments on time, keep your credit card balances low, and manage your debts wisely."
        },
        {
            "question_id": "0f4322b9-b7f2-4b1b-bbd9-92a5b5d1b7d5",
            "question": "Can someone my age have a credit score?",
            "answer": "Yes, once you turn 18 and start using credit products, like a credit card or loan, you begin to build a credit history."
        },
        {
            "question_id": "b18a9e83-43bd-4b78-8f68-1e6c3c5d9b22",
            "question": "What’s on a credit report?",
            "answer": "Details of your credit accounts, payment history, debts, and sometimes employment history, used to calculate your score."
        }
    ]
    score = await qa_evaluator.evaluate_qa_pairs(qa_pairs, model_id)

if __name__ == "__main__":
    asyncio.run(start_evaluation())
