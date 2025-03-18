import asyncio
import json
import traceback
from typing import List, Union

import numpy as np
from tqdm import tqdm
from app.services.pinecone_vectorizer import OpenAIEmbedding, VectorizerEngine
from app.utils.config import settings
import openai
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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
        self.model_id = settings.FINETUNED_MODEL_NAME
        self.openai = openai
        self.openai.api_key = settings.OPENAI_API_KEY
        self.client = self.openai.Client()  # Create a client instance
        self.vectorizer = VectorizerEngine(
            encoder=self.encoder,
            vector_db_name=settings.VECTOR_DB_NAME,
            batch_size=10,
            dimension=768,
            namespace="questions"
        )
        self.similarity_threshold = 0.8

    async def get_precision_recall_f1(self, generated_sim_scores: List[float], true_sim_scores: List[float]) -> Union[dict, None]:
        """
        
        Compute the precision, recall, and F1 score based on the similarity scores

        Args:
            generated_sim_scores (List[float]): List of generated similarity scores
            true_sim_scores (List[float]): List of true similarity scores            

        Returns:
            dict: Average score, standard deviation, coefficient of variation, precision, recall, and F1 score
        """
        try:
            score = {}
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
        except Exception as e:
            print()


    async def get_ai_response(self, question) -> Union[str, None]:
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,  # Correct method for OpenAI >= 1.0
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a Credit Genius Assistant."},
                    {"role": "user", "content": question}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error getting AI response: {e}")
            return None
    
    # async def get_ai_response(self, question) -> Union[str, None]:
    #     try:
    #         response = self.openai.ChatCompletion.create(
    #             model=self.model_id,
    #             messages=[
    #                 {"role": "system", "content": "You are a Genius Assistant."},
    #                 {"role": "user", "content": question}
    #             ],
    #             temperature= self.temperature,
    #             max_tokens=self.max_tokens
    #         )
    #         return response['choices'][0]['message']['content'].strip()
    #     except Exception as e:
    #         print(f"Error getting AI response: {e}")
    #         return None


    async def sync_vector_db(self, qa_pairs: List[dict]) -> bool: 
        """
        Sync the vector DB with the latest data.
        """
        try:
            # Load the vector store
            self.vectorizer.load_vectorstore()
            # Sync the vector DB with the latest QA pairs
            await self.vectorizer.create_vectorstore(qa_pairs)
            return True
        except Exception as e:
            print(f"Error syncing vector DB: {e}")
            return False
        finally:
            # Unload the vector store
            self.vectorizer.unload_vectorstore()
        

    async def evaluate_single_qa(self, question: str) -> float:
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
            ai_response = await self.get_ai_response(question)
            if ai_response:
                # Get the similarity score between the question actual answer and the AI response
                generated_similarity_score, true_similarity_score = await self.vectorizer.get_qa_similarity_score(
                    question=question,
                    answer=ai_response
                )
                return ai_response, round(generated_similarity_score, 3), round(true_similarity_score, 3)
            else:
                return None, 0.0, 0.0
        except Exception as e:
            print(f"Error evaluating single QA: {e}")
            return None, 0.0, 0.0
        finally:
            # Unload the vector store
            self.vectorizer.unload_vectorstore()
    

    async def evaluate_qa_pairs(self, qa_pairs: List[dict]) -> Union[dict, None]:
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
                    generated_answer, generated_score, true_score = await self.evaluate_single_qa(question)
                    generated_score_list.append(generated_score)
                    true_score_list.append(true_score)
                    generated_answers[question] = {"answer": generated_answer, "similarity_score": generated_score}
                agg_scores = await self.get_precision_recall_f1(generated_score_list,true_score_list)
            return {
                "generated_answers": generated_answers,
                "agg_scores": agg_scores
            }
        except Exception as e:
            print(f"Error evaluating QA pairs: {e}")
            print(traceback.format_exc())
            return None