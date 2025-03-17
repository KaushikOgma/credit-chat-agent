import asyncio
from app.services.pinecone_vectorize import OpenAIEmbedding, VectorizerEngine
from app.utils.config import settings
import openai


class SingleQAHandler:
    def __init__(self, model_id):
        # Initialize the OpenAI Embedding
        self.encoder = OpenAIEmbedding(model_name=settings.EMBEDDING_MODEL_NAME)
        # Initialize the Vectorizer Engine
        self.temperature = 0.7
        self.max_tokens = 500
        self.model_id = model_id
        self.openai = openai
        self.openai.api_key = settings.OPENAI_API_KEY
        self.vectorizer = VectorizerEngine(
            encoder=self.encoder,
            vector_db_name=settings.VECTOR_DB_NAME,
            batch_size=10,
            dimension=768,
            namespace="questions"
        )

    async def get_ai_response(self, question):
        try:
            response = self.openai.ChatCompletion.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a Credit Genius Assistant."},
                    {"role": "user", "content": question}
                ],
                temperature= self.temperature,
                max_tokens=self.max_tokens
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error: {e}"

    async def evaluate_single_qa(self, question: str) -> dict:
        """
        Evaluate a single QA by checking if the AI response is present in the vector DB.

        Args:
            question (str): Input question

        Returns:
            dict: Similarity score and matched response from the vector DB.
        """
        try:
            ai_response = await get_ai_response(question)
            # Step 1: Generate the embedding for the input question
            query_embedding = await asyncio.to_thread(self.encoder.embed_query, question)

            similarity_score = await self.vectorizer.get_qa_similarity_score(
                question=question,
                answer=ai_response
            )

            return {
                "question": question,
                "ai_response": ai_response,
                "similarity_score": round(similarity_score, 3)
            }

        except Exception as e:
            return {"error": str(e)}
       