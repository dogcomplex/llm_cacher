# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss
import torch
import os
import json
import threading
import time
import logging
from openai import AsyncOpenAI

# Define constants and configurations
GENERAL_MODEL_NAME = 'gpt2-medium'  # Update the model to a more recent, content-filtered version
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # For embedding generation
CACHE_THRESHOLD = 2  # Number of times a query must appear to be cached

# Paths for storing models and data
CACHE_DIR = 'cache_models'
EMBEDDING_INDEX_PATH = 'embedding_index.faiss'
EMBEDDING_DATA_PATH = 'embedding_data.json'

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, filename='system.log', format='%(asctime)s - %(levelname)s - %(message)s')

# set env inline:
api_key = ""
# Add this to the existing imports and constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or api_key
USE_OPENAI = os.getenv("USE_OPENAI", "true").lower() == "true"

class OpenAIModel:
    """
    Represents the OpenAI GPT-4 model for handling queries via the OpenAI API.
    """
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def generate_response(self, input_text):
        """
        Generate a response using the OpenAI GPT-4 model.

        Args:
            input_text (str): The input text.

        Returns:
            str: The generated response.
        """
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_text}
                ],
                max_tokens=150,
                n=1,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error generating response from OpenAI: {e}")
            return "An error occurred while processing your request with OpenAI."

class QueryProcessor:
    """
    Handles the processing of incoming queries, including preprocessing,
    embedding generation, and passing queries to the appropriate handler.
    """
    def __init__(self, cache_manager, embedding_system):
        self.cache_manager = cache_manager
        self.embedding_system = embedding_system
        self.general_model = GeneralModel()
        self.lock = threading.Lock()  # For thread-safe increments

    def process_query(self, query_text):
        try:
            logging.info(f"Received query: {query_text}")
            # ... (existing code)
            return response
        except Exception as e:
            logging.error(f"Error processing query '{query_text}': {e}")
            return "An error occurred while processing your request."

    def preprocess(self, query_text):
        """
        Preprocess the input query text.

        Args:
            query_text (str): The raw input text.

        Returns:
            str: The preprocessed text.
        """
        # Implement any necessary preprocessing steps
        return query_text.strip().lower()

    def update_query_stats(self, query_text):
        """
        Update the frequency count for the query and initiate caching if threshold is reached.

        Args:
            query_text (str): The preprocessed query text.
        """
        with self.lock:
            frequency = self.cache_manager.increment_query_count(query_text)
            if frequency >= CACHE_THRESHOLD:
                # Initiate model distillation and caching
                self.cache_manager.cache_submodel(query_text)


class GeneralModel:
    """
    Represents the primary generalized AI model for handling uncached queries.
    """
    def __init__(self):
        if USE_OPENAI:
            self.model = OpenAIModel()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(GENERAL_MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(GENERAL_MODEL_NAME)
            
            # Set pad token ID if it's not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id

    async def generate_response(self, input_text):
        """
        Generate a response using either the local model or OpenAI model.

        Args:
            input_text (str): The input text.

        Returns:
            str: The generated response.
        """
        if USE_OPENAI:
            return await self.model.generate_response(input_text)
        else:
            return self._generate_local_response(input_text)

    def _generate_local_response(self, input_text):
        """
        Generate a response using the local AI model.

        Args:
            input_text (str): The input text.

        Returns:
            str: The generated response.
        """
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=100,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


class CacheManager:
    """
    Manages cached specialized submodels, query frequencies, and storage resources.
    """
    def __init__(self, embedding_system, general_model):
        self.embedding_system = embedding_system
        self.general_model = general_model  # Add this line
        self.query_counts = {}  # Dictionary to store query frequencies
        self.lock = threading.Lock()  # Add this for thread safety

    def increment_query_count(self, query_text):
        """
        Increment the frequency count for a given query.

        Args:
            query_text (str): The query text.

        Returns:
            int: The updated frequency count.
        """
        with self.lock:  # Ensure thread-safe access
            self.query_counts[query_text] = self.query_counts.get(query_text, 0) + 1
            return self.query_counts[query_text]

    def cache_submodel(self, query_text):
        """
        Create a specialized submodel for a frequent query and cache it.

        Args:
            query_text (str): The query text to create a submodel for.
        """
        # Distill a specialized model for the query
        submodel = self.distill_model(query_text)

        # Save the specialized model
        model_path = os.path.join(CACHE_DIR, f"model_{hash(query_text)}")
        submodel.save_pretrained(model_path)

        # Update the embedding system with the new cached model
        self.embedding_system.add_cached_model(query_text, model_path)

    def distill_model(self, query_text):
        """
        Distill a smaller, specialized model for the given query.

        Args:
            query_text (str): The query text.

        Returns:
            PreTrainedModel: The distilled specialized model.
        """
        # Initialize the smaller model and tokenizer
        small_model_name = 'distilgpt2'  # A smaller, faster model
        tokenizer = AutoTokenizer.from_pretrained(small_model_name)
        model = AutoModelForCausalLM.from_pretrained(small_model_name)

        # Prepare synthetic training data using the general model
        training_data = self.generate_training_data(query_text)

        # Fine-tune the smaller model on the training data
        self.fine_tune_model(model, tokenizer, training_data)

        return model

    def generate_training_data(self, query_text):
        """
        Generate synthetic training data for distillation.

        Args:
            query_text (str): The query text.

        Returns:
            list of dict: A list containing input-output pairs.
        """
        inputs = [query_text] * 100  # Generate multiple instances
        responses = []

        for input_text in inputs:
            response = self.general_model.generate_response(input_text)
            responses.append({'input': input_text, 'output': response})

        return responses

    def fine_tune_model(self, model, tokenizer, training_data):
        """
        Fine-tune the model on the given training data.

        Args:
            model (PreTrainedModel): The model to fine-tune.
            tokenizer (PreTrainedTokenizer): The tokenizer.
            training_data (list of dict): The training data.
        """
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import Trainer, TrainingArguments

        class CustomDataset(Dataset):
            def __init__(self, data, tokenizer):
                self.data = data
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                input_ids = self.tokenizer.encode(item['input'], return_tensors='pt').squeeze()
                labels = self.tokenizer.encode(item['output'], return_tensors='pt').squeeze()
                return {'input_ids': input_ids, 'labels': labels}

        dataset = CustomDataset(training_data, tokenizer)

        training_args = TrainingArguments(
            output_dir='./fine_tuned_models',
            num_train_epochs=1,
            per_device_train_batch_size=2,
            save_steps=5000,
            save_total_limit=2,
            logging_steps=500,
            logging_dir='./logs',
            learning_rate=5e-5,
            weight_decay=0.01
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=lambda data: {'input_ids': torch.nn.utils.rnn.pad_sequence([f['input_ids'] for f in data], batch_first=True, padding_value=tokenizer.pad_token_id),
                                        'labels': torch.nn.utils.rnn.pad_sequence([f['labels'] for f in data], batch_first=True, padding_value=-100)}
        )

        trainer.train()
    def generate_response(self, model_path, input_text):
        """
        Generate a response using a cached specialized submodel.

        Args:
            model_path (str): Path to the specialized submodel.
            input_text (str): The input text.

        Returns:
            str: The generated response.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        inputs = tokenizer.encode(input_text, return_tensors='pt')
        outputs = model.generate(inputs, max_length=100, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


class EmbeddingRetrievalSystem:
    """
    Handles embedding generation and retrieval of cached models based on query similarity.
    """
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = self._load_or_create_index()
        self.model_paths = self._load_model_paths()

    def _load_or_create_index(self):
        """
        Load the FAISS index or create a new one if it doesn't exist.

        Returns:
            faiss.IndexFlatL2: The FAISS index for embeddings.
        """
        if os.path.exists(EMBEDDING_INDEX_PATH):
            index = faiss.read_index(EMBEDDING_INDEX_PATH)
        else:
            index = faiss.IndexFlatL2(self.dimension)
        return index

    def _load_model_paths(self):
        """
        Load the mapping of embeddings to model paths.

        Returns:
            dict: A dictionary mapping from index IDs to model paths.
        """
        if os.path.exists(EMBEDDING_DATA_PATH):
            with open(EMBEDDING_DATA_PATH, 'r') as f:
                model_paths = json.load(f)
        else:
            model_paths = {}
        return model_paths

    def generate_embedding(self, texts):
        """
        Generate embeddings for the given texts.

        Args:
            texts (list of str): The input texts.

        Returns:
            np.ndarray: The embeddings matrix.
        """
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, batch_size=32)
        return embeddings

    def retrieve_cached_model(self, embedding):
        """
        Retrieve the cached model path for a given embedding if it exists.

        Args:
            embedding (np.ndarray): The query embedding.

        Returns:
            str or None: The path to the cached model or None if not found.
        """
        if self.index.ntotal == 0:
            return None

        k = 1  # Number of nearest neighbors to retrieve
        distances, indices = self.index.search(embedding.reshape(1, -1), k)
        closest_distance = distances[0][0]
        closest_index = indices[0][0]

        # Define a similarity threshold
        SIMILARITY_THRESHOLD = 0.5  # Adjust based on experimentation

        if closest_distance < SIMILARITY_THRESHOLD:
            model_path = self.model_paths.get(str(closest_index))
            return model_path
        else:
            return None

    def add_cached_model(self, query_text, model_path):
        """
        Add a new cached model to the embedding index and model paths.

        Args:
            query_text (str): The query text associated with the model.
            model_path (str): The file path to the cached model.
        """
        # Generate embedding for the query text
        embedding = self.generate_embedding(query_text)

        # Add embedding to the index
        self.index.add(embedding.reshape(1, -1))

        # Update the model paths mapping
        index_id = self.index.ntotal - 1
        self.model_paths[str(index_id)] = model_path

        # Save the updated index and model paths
        faiss.write_index(self.index, EMBEDDING_INDEX_PATH)
        with open(EMBEDDING_DATA_PATH, 'w') as f:
            json.dump(self.model_paths, f)

class ReinforcementLearningAgent:
    """
    Optimizes caching strategies over time based on system performance metrics.
    """
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.model_values = {}  # Stores value estimates for cached models

    def update_policy(self):
        """
        Update the caching policy based on reinforcement learning.
        """
        for model_path, stats in self.cache_manager.model_stats.items():
            reward = self.calculate_reward(stats)
            self.model_values[model_path] = self.model_values.get(model_path, 0) + reward

        self.optimize_cache()

    def calculate_reward(self, stats):
        """
        Calculate the reward for a cached model.

        Args:
            stats (dict): Usage statistics for the model.

        Returns:
            float: The calculated reward.
        """
        usage = stats['usage_count']
        latency = stats['average_response_time']
        size = stats['model_size']

        # Reward function can be customized
        reward = usage / (latency * size)
        return reward

    def optimize_cache(self):
        """
        Remove models with the lowest value estimates to manage storage.
        """
        sorted_models = sorted(self.model_values.items(), key=lambda item: item[1])

        while self.cache_manager.storage_manager.current_storage > self.cache_manager.storage_manager.max_storage:
            model_to_remove, _ = sorted_models.pop(0)
            self.cache_manager.remove_cached_model(model_to_remove)
            del self.model_values[model_to_remove]


class DataStorageManager:
    """
    Manages disk space utilization, serialization, and deserialization of models.
    """
    def __init__(self, max_storage=100 * 1024 * 1024 * 1024):  # 100 GB
        self.max_storage = max_storage
        self.current_storage = self.calculate_current_storage()

    def calculate_current_storage(self):
        """
        Calculate the current storage used by cached models.

        Returns:
            int: The total size of cached models in bytes.
        """
        total_size = 0
        for root, dirs, files in os.walk(CACHE_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        return total_size

    def manage_storage(self):
        """
        Manage the storage by deleting less-used models.
        """
        if self.current_storage > self.max_storage:
            logging.info("Storage limit exceeded. Initiating cache optimization.")
            self.cache_manager.rl_agent.update_policy()
            self.current_storage = self.calculate_current_storage()
        else:
            logging.info("Storage usage within limits.")

    def get_available_storage(self):
        """
        Calculate the available storage space.

        Returns:
            int: The available storage in bytes.
        """
        return self.max_storage - self.current_storage


import asyncio

class AsyncQueryProcessor(QueryProcessor):
    async def process_query(self, query_text):
        try:
            logging.info(f"Received query: {query_text}")
            preprocessed_query = self.preprocess(query_text)
            
            # Use asyncio.to_thread for CPU-bound operations
            embedding = await asyncio.to_thread(self.embedding_system.generate_embedding, [preprocessed_query])
            cached_model_path = await asyncio.to_thread(self.embedding_system.retrieve_cached_model, embedding[0])

            if cached_model_path:
                response = await asyncio.to_thread(self.cache_manager.generate_response, cached_model_path, preprocessed_query)
            else:
                response = await self.general_model.generate_response(preprocessed_query)

            await asyncio.to_thread(self.update_query_stats, preprocessed_query)
            return response
        except Exception as e:
            logging.error(f"Error processing query '{query_text}': {e}")
            return "An error occurred while processing your request."

# Usage example
async def main():
    embedding_system = EmbeddingRetrievalSystem()
    general_model = GeneralModel()
    cache_manager = CacheManager(embedding_system, general_model)
    async_query_processor = AsyncQueryProcessor(cache_manager, embedding_system)
    rl_agent = ReinforcementLearningAgent(cache_manager)
    storage_manager = DataStorageManager()

    # Add the storage manager to the cache manager
    cache_manager.storage_manager = storage_manager

    # Define queries here
    queries = [
        "How do I reset my password?",
        "What's the weather like today?",
        "How do I reset my password?",
        "Tell me a family-friendly joke.",
        "How do I reset my password?",
        # Add more queries as needed
    ]

    tasks = [async_query_processor.process_query(query) for query in queries]
    responses = await asyncio.gather(*tasks)
    for query, response in zip(queries, responses):
        print(f"Query: {query}\nResponse: {response}\n")

if __name__ == '__main__':
    asyncio.run(main())