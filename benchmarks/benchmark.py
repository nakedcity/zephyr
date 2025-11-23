import asyncio
import httpx
import time
import statistics
import random

SERVER_URL = "http://localhost:8080/v1/embeddings"
MODELS = ["bge-small-en-v1.5", "all-MiniLM-L6-v2"]
NUM_REQUESTS = 50
CONCURRENCY = 10

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "Quantization reduces model size and improves inference speed.",
    "ONNX Runtime provides a high-performance inference engine.",
    "Python is a versatile programming language for data science.",
    "FastAPI is a modern, fast (high-performance), web framework for building APIs.",
    "Machine learning models require significant computational resources.",
    "Edge computing brings computation and data storage closer to the sources of data.",
    "Cloud computing provides on-demand availability of computer system resources.",
    "Deep learning is part of a broader family of machine learning methods."
]

async def send_request(client, model, request_id):
    text = random.choice(SENTENCES)
    payload = {
        "input": text,
        "model": model
    }
    
    start_time = time.time()
    try:
        response = await client.post(SERVER_URL, json=payload, timeout=30.0)
        response.raise_for_status()
        latency = (time.time() - start_time) * 1000  # ms
        return latency, None
    except Exception as e:
        return None, str(e)

async def benchmark_model(model_name):
    print(f"\nBenchmarking {model_name}...")
    print(f"Requests: {NUM_REQUESTS}, Concurrency: {CONCURRENCY}")
    
    async with httpx.AsyncClient() as client:
        tasks = []
        latencies = []
        errors = []
        
        start_total = time.time()
        
        # Create batches of tasks to respect concurrency
        for i in range(0, NUM_REQUESTS, CONCURRENCY):
            batch_tasks = []
            for j in range(CONCURRENCY):
                if i + j < NUM_REQUESTS:
                    batch_tasks.append(send_request(client, model_name, i + j))
            
            results = await asyncio.gather(*batch_tasks)
            
            for lat, err in results:
                if lat is not None:
                    latencies.append(lat)
                else:
                    errors.append(err)
                    
        total_time = time.time() - start_total
        
    if not latencies:
        print("All requests failed!")
        return

    print(f"Total Time: {total_time:.2f}s")
    print(f"Throughput: {len(latencies) / total_time:.2f} req/s")
    print(f"Avg Latency: {statistics.mean(latencies):.2f} ms")
    print(f"P50 Latency: {statistics.median(latencies):.2f} ms")
    print(f"P95 Latency: {statistics.quantiles(latencies, n=20)[18]:.2f} ms")
    print(f"P99 Latency: {statistics.quantiles(latencies, n=100)[98]:.2f} ms")
    if errors:
        print(f"Errors: {len(errors)}")
        print(f"Sample Error: {errors[0]}")

async def main():
    # Warmup
    print("Warming up...")
    async with httpx.AsyncClient() as client:
        for model in MODELS:
            await client.post(SERVER_URL, json={"input": "warmup", "model": model})

    for model in MODELS:
        await benchmark_model(model)

if __name__ == "__main__":
    asyncio.run(main())
