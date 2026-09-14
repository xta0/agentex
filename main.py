from litellm import acompletion
from dotenv import load_dotenv
# from pydantic import BaseModel
import asyncio

load_dotenv()

# class ExtractedInfo(BaseModel):
#     name: str
#     email: str
#     phone: str | None = None

semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests

async def call_llm(prompt: str) -> str:
    """LLM call with rate limiting and automatic retry"""
    async with semaphore:
        response = await acompletion(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            num_retries=3,  # Automatically retry up to 3 times on failure
    )
    return response.choices[0].message.content


async def main():
    prompts = [f"what is {i} + {i}?" for i in range(10)] 
    # execute all requests concurrently
    tasks = [call_llm(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    for prompt, result in zip(prompts, results):
        print(f"Q: {prompt}")
        print(f"A: {result}")


if __name__ == "__main__":
    asyncio.run(main())