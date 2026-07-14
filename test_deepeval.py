import os
import asyncio
from dotenv import load_dotenv
from groq import Groq
from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from bot import get_bot_response

# Load environment variables
load_dotenv()

# Define the custom Groq LLM wrapper for DeepEval
class GroqLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.client = Groq()

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)

    def get_model_name(self) -> str:
        return self.model_name

def run_deepeval_demo():
    print("\n" + "="*50)
    print("STARTING DEEPEVAL EVALUATION")
    print("="*50 + "\n")

    # 1. Initialize the custom Groq evaluator model
    eval_model = GroqLLM(model_name="llama-3.3-70b-versatile")

    # 2. Define the metrics to evaluate
    # We pass our custom Groq model to bypass DeepEval's default OpenAI model requirement
    relevancy_metric = AnswerRelevancyMetric(threshold=0.6, model=eval_model)
    faithfulness_metric = FaithfulnessMetric(threshold=0.6, model=eval_model)

    # 3. Define test inputs (queries)
    test_queries = [
        "How much is premium membership and what features are included?",
        "Can you write a custom Python script to calculate my daily caloric intake?"
    ]

    # 4. Generate bot outputs and construct DeepEval Test Cases
    test_cases = []
    for query in test_queries:
        print(f"Generating bot response for query: '{query}'...")
        bot_res = get_bot_response(query)
        
        # Create LLMTestCase
        test_case = LLMTestCase(
            input=query,
            actual_output=bot_res["response"],
            retrieval_context=[bot_res["context"]]
        )
        test_cases.append(test_case)

    # 5. Measure each test case individually and print results
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Evaluating Test Case {i+1} ---")
        print(f"Query (Input): {test_case.input}")
        print(f"Bot Output: {test_case.actual_output}")
        print(f"Retrieved Context: {test_case.retrieval_context[0]}\n")

        # Measure Answer Relevancy
        print("Measuring Answer Relevancy...")
        relevancy_metric.measure(test_case)
        print(f"-> Relevancy Score: {relevancy_metric.score:.2f} (Passed: {relevancy_metric.is_successful()})")
        print(f"-> Reason: {relevancy_metric.reason}")

        # Measure Faithfulness (RAG accuracy)
        print("\nMeasuring Faithfulness (Grounding)...")
        faithfulness_metric.measure(test_case)
        print(f"-> Faithfulness Score: {faithfulness_metric.score:.2f} (Passed: {faithfulness_metric.is_successful()})")
        print(f"-> Reason: {faithfulness_metric.reason}")
        print("-"*40)

if __name__ == "__main__":
    run_deepeval_demo()
