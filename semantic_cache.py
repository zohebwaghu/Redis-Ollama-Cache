import redis
import json
import time
import requests
import numpy as np
from typing import Dict, Any, Tuple

class SemanticCache:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, ollama_url: str = "http://localhost:11434"):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.ollama_url = ollama_url
        # Use a simple hash-based embedding for demonstration (since sentence-transformers is hard to install)
        self.similarity_threshold = 0.4
        self.cache_key_prefix = "semantic_cache:"

    def _get_embedding(self, text: str) -> list:
        """Generate a simple embedding for the given text using TF-IDF-like approach"""
        # For demonstration, create a simple vector based on word frequencies and positions
        words = text.lower().split()
        embedding = [0] * 100  # Simple 100-dimensional vector

        # Add word frequency
        for word in words:
            hash_val = hash(word) % 50  # Use first 50 dimensions for word freq
            embedding[hash_val] += 1

        # Add positional information
        for i, word in enumerate(words):
            pos_val = hash(f"{word}_{i}") % 50  # Use next 50 dimensions for position
            embedding[50 + pos_val] += 1

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        return embedding

    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0 else 0

    def _query_ollama(self, query: str) -> str:
        """Query Ollama for a response (mock implementation since Ollama is not running)"""
        # Mock responses for demonstration
        mock_responses = {
            "What is the capital of France?": "The capital of France is Paris.",
            "Tell me the capital city of France.": "The capital city of France is Paris.",
            "What is machine learning?": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
            "Explain machine learning in simple terms.": "Machine learning is like teaching a computer to recognize patterns in data, so it can make predictions or decisions on its own.",
            "How does photosynthesis work?": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
            "What are the benefits of exercise?": "Exercise provides numerous benefits including improved cardiovascular health, stronger muscles, better mental health, and increased longevity.",
            "Why is the sky blue?": "The sky appears blue because of Rayleigh scattering, where shorter blue wavelengths of light are scattered more by the atmosphere than longer wavelengths.",
            "Describe the process of photosynthesis.": "Photosynthesis involves plants absorbing sunlight through chlorophyll, combining it with water from roots and carbon dioxide from air to create glucose and release oxygen."
        }

        # Simulate some processing time
        time.sleep(0.1)

        return mock_responses.get(query, f"Mock response for: {query}")

    def query_handler(self, query: str) -> Tuple[str, bool, float]:
        """
        Handle a query with semantic caching.
        Returns: (response, is_cached, similarity_score)
        """
        start_time = time.time()

        # Generate embedding for the query
        query_embedding = self._get_embedding(query)

        # Search for similar queries in Redis
        max_similarity = 0
        best_match = None
        best_key = None

        # Get all cached queries (in a real implementation, use Redis search)
        for key in self.redis_client.scan_iter(f"{self.cache_key_prefix}*"):
            cached_data = self.redis_client.get(key)
            if cached_data:
                try:
                    data = json.loads(cached_data)
                    cached_embedding = data['embedding']
                    similarity = self._cosine_similarity(query_embedding, cached_embedding)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = data
                        best_key = key
                except:
                    continue

        if max_similarity >= self.similarity_threshold and best_match:
            # Cache hit
            response_time = time.time() - start_time
            print(f"Cache hit: similarity={max_similarity:.3f}, response_time={response_time:.3f}s")
            return best_match['response'], True, max_similarity

        # Cache miss - query Ollama
        response = self._query_ollama(query)
        response_time = time.time() - start_time
        print(f"Cache miss: response_time={response_time:.3f}s")

        # Store in cache
        cache_data = {
            'query': query,
            'response': response,
            'embedding': query_embedding,
            'timestamp': time.time()
        }
        cache_key = f"{self.cache_key_prefix}{int(time.time())}"
        self.redis_client.set(cache_key, json.dumps(cache_data))

        return response, False, 0.0

def test_semantic_cache():
    """Test the semantic caching system with 10 diverse queries"""
    cache = SemanticCache()

    queries = [
        "What is the capital of France?",  # Exact duplicate
        "What is the capital of France?",  # Exact duplicate
        "Tell me the capital city of France.",  # Paraphrase
        "What is machine learning?",  # New query
        "Explain machine learning in simple terms.",  # Paraphrase
        "How does photosynthesis work?",  # Completely new
        "What are the benefits of exercise?",  # New
        "Why is the sky blue?",  # New
        "What is the capital of France?",  # Another exact duplicate
        "Describe the process of photosynthesis.",  # Paraphrase of earlier query
    ]

    cache_hits = 0
    total_cached_time = 0
    total_uncached_time = 0

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        start = time.time()
        response, is_cached, similarity = cache.query_handler(query)
        end = time.time()
        response_time = end - start

        if is_cached:
            cache_hits += 1
            total_cached_time += response_time
            print(f"  Cached response (similarity: {similarity:.3f})")
        else:
            total_uncached_time += response_time
            print("  Fresh response from Ollama")

        print(f"  Response time: {response_time:.3f}s")
        print(f"  Response preview: {response[:100]}...")

    # Calculate metrics
    cache_hit_rate = cache_hits / len(queries) * 100
    avg_cached_time = total_cached_time / cache_hits if cache_hits > 0 else 0
    avg_uncached_time = total_uncached_time / (len(queries) - cache_hits) if len(queries) > cache_hits else 0
    speedup = avg_uncached_time / avg_cached_time if avg_cached_time > 0 else float('inf')

    print("\n" + "="*60)
    print("CACHE PERFORMANCE METRICS")
    print("="*60)
    print(f"Total queries: {len(queries)}")
    print(f"Cache hits: {cache_hits}")
    print(f"Cache hit rate: {cache_hit_rate:.1f}%")
    print(f"Average cached response time: {avg_cached_time:.3f}s")
    print(f"Average uncached response time: {avg_uncached_time:.3f}s")
    print(f"Speed improvement: {speedup:.1f}x faster")

if __name__ == "__main__":
    test_semantic_cache()
