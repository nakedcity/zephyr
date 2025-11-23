import asyncio
import httpx
import time
import statistics

SERVER_URL = "http://localhost:8080/v1/embeddings"
MODEL = "bge-small-en-v1.5"  # GPU model
NUM_REQUESTS = 100
CONCURRENCY = 20

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "Quantization reduces model size and improves inference speed.",
    "ONNX Runtime provides a high-performance inference engine.",
    "Python is a versatile programming language for data science.",
]

async def send_request(client, request_id):
    text = SENTENCES[request_id % len(SENTENCES)]
    payload = {"input": text, "model": MODEL}
    
    start_time = time.time()
    try:
        response = await client.post(SERVER_URL, json=payload, timeout=30.0)
        response.raise_for_status()
        latency = (time.time() - start_time) * 1000
        return latency, None
    except Exception as e:
        return None, str(e)

async def main():
    print(f"Benchmarking {MODEL} on GPU...")
    print(f"Requests: {NUM_REQUESTS}, Concurrency: {CONCURRENCY}")
    
    async with httpx.AsyncClient() as client:
        latencies = []
        errors = []
        
        start_total = time.time()
        
        for i in range(0, NUM_REQUESTS, CONCURRENCY):
            batch_tasks = []
            for j in range(CONCURRENCY):
                if i + j < NUM_REQUESTS:
                    batch_tasks.append(send_request(client, i + j))
            
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

    print(f"\nResults:")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Throughput: {len(latencies) / total_time:.2f} req/s")
    print(f"Avg Latency: {statistics.mean(latencies):.2f} ms")
    print(f"P50 Latency: {statistics.median(latencies):.2f} ms")
    print(f"P95 Latency: {statistics.quantiles(latencies, n=20)[18]:.2f} ms")
    print(f"P99 Latency: {statistics.quantiles(latencies, n=100)[98]:.2f} ms")
    if errors:
        print(f"Errors: {len(errors)}")

if __name__ == "__main__":
    asyncio.run(main())
