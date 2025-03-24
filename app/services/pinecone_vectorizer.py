"""
This module provides functions and classes for handling various data processing tasks
using libraries such as NumPy, Pandas, and LangChain. It includes utilities for working
with language models (encoders), managing document stores, and integrating with Chroma for efficient
document retrieval.
"""

import asyncio
import os
import json
import gc
import time
from typing import List, Optional, Dict, Tuple, Any
from uuid import UUID
import uuid
from tqdm import tqdm
import numpy as np
import pandas as pd
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from langchain_openai import OpenAIEmbeddings
from app.utils.config import settings
import openai
from langchain_pinecone import PineconeVectorStore
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from langchain.docstore.document import Document
from app.utils.helpers.common_helper import preprocess_text
from app.utils.logger import setup_logger
logger = setup_logger()


class OpenAIEmbedding:
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        """Initialize OpenAI embedding model."""
        openai.api_key = settings.OPENAI_API_KEY
        self.model = OpenAIEmbeddings(model=model_name)
        self.service_name = "pinecone_embedder"

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents.

        Args:
            documents (List[str]): List of documents to embed

        Returns:
            List[List[float]]: List of embeddings
        """
        try:
            return self.model.embed_documents(documents)
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query.

        Args:
            query (str): Query to generate embedding for

        Returns:
            List[float]: Embedding vector
        """
        try:
            return self.model.embed_query(query)
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error


class VectorizerEngine:
    """
    A class to handle vectorization and document storage operations.
    """

    def __init__(
        self,
        encoder: OpenAIEmbeddings,
        vector_db_name: str,
        batch_size: int = 128,
        dimension: int = 768,
        namespace: str = "questions",
    ):
        """
        Initialize the Recommendation Engine.

        Args:
            data (pd.DataFrame): input dataframe
            gguf_threads (int): Number of threads to use to run inference using GGUF
            vector_db_persist_dir (str): Persist dir for Vector store.
            batch_size (int, optional): Batch size for computing embedding. Defaults to 128.
        """

        self.vectordb = None
        self.batch_size = batch_size
        self.vector_db_name = vector_db_name
        self.dimension = dimension
        self.namespace = namespace
        self.semaphore = asyncio.Semaphore(settings.MAX_THREADS)
        self.encoder = encoder
        self.service_name = "pinecone_vectorizer"

    @staticmethod
    def process_chunk(
        qa_pairs: List[dict],
    ) -> Tuple[List[Document], List[UUID]]:
        try:

            # DataFrame creation
            qa_pair_df = pd.DataFrame(qa_pairs)

            # Create corpus and drop temporary column
            corpus = qa_pair_df["question"].tolist()

            # Collect food IDs and metadata
            chunk_qa_pair_id_list = qa_pair_df["question_id"].tolist()
            metadata = qa_pair_df.to_dict(orient="records")

            # Build Document instances
            chunk_documents = []
            for text_item, meta in zip(corpus, metadata):
                chunk_documents.append(Document(page_content=preprocess_text(text_item), metadata=meta))

            return chunk_documents, chunk_qa_pair_id_list

        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return [], []

    async def generate_documents(
        self,
        data: List[Dict],
    ) -> Tuple[List[Document], List[UUID]]:
        """
        Generate documents from a given dataset.

        Args:
            data (List[Dict]): The dataset to generate documents from.

        Returns:
            Tuple[List[List], List[List]]: A tuple of two lists. The first list contains the generated documents, and the second list contains the IDs of the foods in the dataset.

        Raises:
            Exception: If an error occurs while generating documents.
        """
        try:
            documents = []
            foodIdList = []
            chunk_size = 100
            loop = asyncio.get_running_loop()
            # Split data into chunks and process each chunk in parallel
            if loop.is_running():
                with ProcessPoolExecutor(
                    max_workers=settings.MAX_PROCESSES,
                    max_tasks_per_child=5,  # adjust_max_tasks_per_child(),
                ) as process_executor:
                    try:
                        # Run in the shared process pool
                        tasks = [
                            loop.run_in_executor(
                                process_executor,
                                VectorizerEngine.process_chunk,
                                data[i : i + chunk_size],
                            )
                            for i in range(0, len(data), chunk_size)
                        ]

                        # Gather all tasks concurrently
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        for result in results:
                            try:
                                chunk_documents, chunk_foodIds = result
                                documents.extend(chunk_documents)
                                foodIdList.extend(chunk_foodIds)
                            except Exception as e:
                                print(result)
                                print(f"Error while processing chunk: {e}")
                    except Exception as e:
                        print(f"generate_documents:: Error in async processing: {e}")
                    finally:
                        process_executor.shutdown(wait=True)
            return documents, foodIdList
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return [], []

    def split_into_batches(
        self, ids, documents, batch_size
    ) -> Tuple[List[Document], List[UUID]]:
        """Splits the data into batches of given batch size.

        Args:
            ids (List): List of identifiers.
            documents (List): List of documents corresponding to the ids.
            batch_size (int): The desired batch size.

        Returns:
            Optional[Tuple[List[List], List[List]]]: A tuple containing two lists:
                - List of document batches
                - List of id batches
            Returns None if there is an error.
        """
        try:
            if len(documents) != len(ids):
                raise ValueError("Documents and IDs should have the same length")

            batches = [
                (documents[i : i + batch_size], ids[i : i + batch_size])
                for i in range(0, len(documents), batch_size)
            ]

            document_batches, id_batches = zip(*batches)

            return list(document_batches), list(id_batches)

        except ValueError as ve:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return [], []
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return [], []

    async def add_batch_in_vectordb(
        self, data, ids
    ) -> bool:
        """
        Add a batch of documents to the vector database.

        Args:
            data (List[Document]): List of Document objects to add.
            ids (List): List of identifiers corresponding to the documents.

        Returns:
            bool: True if successful, False if an error occurs.
        """
        try:
            text_batch = []
            metadata_batch = []
            for doc in data:
                text_batch.append(doc.page_content)
                metadata_batch.append(doc.metadata)
            # embeddings = self.encoder.embed_query(text_batch)
            embeddings = await asyncio.to_thread(self.encoder.embed_documents, text_batch)
            upsert_data = [
                (id, embedding, metadata)
                for id, embedding, metadata in zip(ids, embeddings, metadata_batch)
            ]
            # self.vectordb.upsert(
            #     upsert_data, namespace=self.namespace, show_progress=True
            # )
            await asyncio.to_thread(
                self.vectordb.upsert,
                upsert_data,
                self.namespace,
                True,
                show_progress=False,
            )
            return True
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return False

    async def process_batch(self, document_batch, id_batch) -> bool:
        """Helper coroutine to process a single batch with retries."""
        try:
            max_retry = 3
            current_try = 0
            while current_try <= max_retry:
                if current_try > 0:
                    print(f"Batch Insert Retrying count: {current_try}")
                    await asyncio.to_thread(gc.collect)
                    await asyncio.sleep(3)  # Delay before retrying

                current_try += 1
                # Acquire semaphore to limit the number of concurrent tasks
                async with self.semaphore:
                    # Attempt to add the batch to vector database
                    if await self.add_batch_in_vectordb(document_batch, id_batch):
                        return True  # Successfully inserted

            return False  # Failed after retries
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return False

    async def init_vectorstore(
        self, ids, documents, batch_size: int
    ) -> Optional[PineconeVectorStore]:
        """
        Initializes the vector store by processing and ingesting document batches.

        Args:
            ids (List): List of identifiers for the documents.
            documents (List): List of documents to be processed.
            batch_size (int): The size of each batch for processing.

        Returns:
            Optional[PineconeVectorStore]: Returns the initialized vector store if successful, else returns False.

        Raises:
            Exception: If an error occurs during the initialization process.
        """
        try:
            # Generate mini batches
            document_batches, id_batches = self.split_into_batches(
                ids, documents, batch_size
            )
            tasks = [
                self.process_batch(document_batch, id_batch)
                for document_batch, id_batch in zip(document_batches, id_batches)
            ]

            # Gather tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return all(results)
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return False

    async def create_vectorstore(
        self,
        data: List[Dict]
    ) -> bool:
        """
        Create a vector store from a given dataset.

        Args:
            data (List[Dict]): The dataset to create the vector store from.

        Returns:
            Boolean: True if the vector store is successfully created, False otherwise.
        """
        try:
            self.load_vectorstore()
            # print("Loaded vector store")
            # Create documents are IDs
            documents, ids = await self.generate_documents(
                data
            )
            print(f"Length of ids: {len(ids)}")
            if len(ids):
                await self.init_vectorstore(
                    ids=ids, documents=documents, batch_size=self.batch_size
                )
                # self.vectordb.delete(ids=toBeDeletedFoodIds)
            documents = None
            ids = None
            # self.unload_vectorstore()
            # print("Unloaded vector store")
            return True
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return False

    async def delete_item_chunk(
        self, chunk, retry_count=0, delay=2, max_retries=3
    ) -> bool:
        try:
            """Deletes a chunk of food items with retry logic."""
            for attempt in range(max_retries):
                try:
                    self.load_vectorstore()  # Ensure vectorstore is loaded
                    await asyncio.to_thread(
                        lambda: self.vectordb.delete(
                            ids=chunk, namespace=self.namespace
                        )
                    )
                    return True  # Success
                except Exception as error:
                    print(
                        f"delete_item_chunk: Chunk delete error: {error} - Attempt {attempt + 1}"
                    )
                    print("delete_item_chunk traceback: ", traceback.format_exc())
                    if attempt < max_retries - 1:  # Avoid delay on the last attempt
                        await asyncio.sleep(delay)  # Async wait for exponential backoff
                        delay *= 2  # Exponential backoff
            return False  # All retries failed
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error

    async def delete_items(self, id_list: List[str]) -> bool:
        """
        Deletes the specified food items from the vector database.

        Args:
            id_list (List[str]): A list of food IDs to be deleted.

        Returns:
            bool: True if the food items are successfully deleted, False otherwise.

        Retries the deletion process up to a maximum of `max_retries` times with
        exponential backoff delay if an exception occurs during the deletion.
        """
        try:
            retry_count = 0
            max_retries = 3
            initial_delay = 2  # Initial delay for retry logic (in seconds)
            chunk_size = 100
            # Split id_list into chunks for parallel processing
            id_chunks = [
                id_list[i : i + chunk_size]
                for i in range(0, len(id_list), chunk_size)
            ]
            # Prepare async tasks for each chunk
            tasks = [
                self.delete_item_chunk(chunk, retry_count, initial_delay, max_retries)
                for chunk in id_chunks
            ]
            # Run tasks concurrently and collect results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return all(results)  # Returns True if all chunks are successfully deleted
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error

    def load_vectorstore(self) -> None:
        """
        Load existing vector store
        """
        try:
            self.vectordb = PineconeVectorStore(
                index_name=self.vector_db_name,
                embedding=self.encoder,
                distance_strategy="COSINE",
            ).get_pinecone_index(self.vector_db_name)
            # print("loaded vector store")
            return True
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error

    def unload_vectorstore(self) -> None:
        """
        Unload existing vector store
        """
        try:
            self.vectordb = None
            return True
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error

    async def cosine_similarity(self, X: np.ndarray, Y: np.ndarray, batch_size: int = 1000, n_jobs: int = -1) -> np.ndarray:
        """
        Efficient cosine similarity using scipy with parallel execution for large batches.
        """
        try:
            # Ensure both X and Y are 2D
            X = X.reshape(1, -1) if X.ndim == 1 else X
            Y = Y.reshape(1, -1) if Y.ndim == 1 else Y
            def compute_batch(X_batch: np.ndarray) -> np.ndarray:
                return 1 - cdist(X_batch, Y, metric='cosine')
            # Split the data into smaller batches
            X_batches = np.array_split(X, max(1, len(X) // batch_size))

            # Parallel execution using joblib
            similarity_batches = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(compute_batch)(X_batch) for X_batch in X_batches
            )
            # Combine the results
            return np.vstack(similarity_batches).item() * 100
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error

    async def get_data_by_ids(self, question_ids: List[str]) -> Optional[Dict]:
        try:
            target_item_vector_data = self.vectordb.fetch(
                ids=question_ids, namespace=self.namespace
            )
            return target_item_vector_data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error

    async def get_qa_similarity_score(
        self,
        question: str,
        answer: str
    ):
        """
        Given a question and an answer, compute the similarity score between them.

        Args:
            question (str): The question to compare.
            answer (str): The answer to compare.

        Returns:
            float: The similarity score between the question and the answer.
        """
        try:
            cleaned_question = preprocess_text(question)
            question_embedding = await asyncio.to_thread(self.encoder.embed_query, cleaned_question)
            cleaned_answer = preprocess_text(answer)
            answer_embedding = await asyncio.to_thread(self.encoder.embed_query, cleaned_answer)
            matched_data = await asyncio.to_thread(
                lambda: self.vectordb.query(
                    namespace=self.namespace,
                    vector=question_embedding,
                    top_k=1,
                    include_metadata=True,
                    include_values=True,
                )["matches"]
            )
            if not matched_data:
                print("No relevant match found in vector database.")
                return 0.0, 0.0  # Or any appropriate fallback score
            else:
                matched_answer = matched_data[0]["metadata"].get("answer")
                cleaned_matched_answer = preprocess_text(matched_answer)
                matched_question_embedding = matched_data[0].get("values", question_embedding)
                matched_answer_embedding = await asyncio.to_thread(self.encoder.embed_query, cleaned_matched_answer)

                # Compute similarity scores between the matched answer and the actual answer
                answer_similarity_score = await self.cosine_similarity(np.array(matched_answer_embedding), np.array(answer_embedding))
                # Compute similarity scores between the matched question and the matched answer
                true_similarity_score = await self.cosine_similarity(np.array(matched_question_embedding), np.array(matched_answer_embedding))
                return round(answer_similarity_score, 3), round(true_similarity_score, 3)
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error


openai_embedding_encoder = OpenAIEmbedding(
    model_name=settings.EMBEDDING_MODEL_NAME,
)

# # Define the engine
# vectorizer_engine = VectorizerEngine(
#     encoder=openai_embedding_encoder,
#     vector_db_name=settings.VECTOR_DB_NAME,
#     batch_size=10,
# )
