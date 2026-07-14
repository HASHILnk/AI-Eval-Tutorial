import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, HALLUCINATION_PROMPT
from bot import get_bot_response

# Load environment variables
load_dotenv()

def run_openevals_demo():
    print("\n" + "="*50)
    print("STARTING OPENEVALS EVALUATION")
    print("="*50 + "\n")

    # 1. Initialize the ChatGroq model to serve as the Judge
    # OpenEvals uses LangChain ChatModels directly
    groq_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0
    )

    # 2. Create the evaluators
    # We pass the pre-built prompts and specify the groq_model as the judge
    correctness_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        judge=groq_model
    )

    hallucination_evaluator = create_llm_as_judge(
        prompt=HALLUCINATION_PROMPT,
        feedback_key="hallucination",
        judge=groq_model
    )

    # 3. Define test scenarios
    # OpenEvals requires context and inputs. We'll use the same queries as DeepEval
    test_scenarios = [
        {
            "query": "How much is premium membership and what features are included?",
            "expected_answer": "Premium tier costs $9.99/month and includes unlimited workouts, personalized diet plans, offline access, and an ad-free experience."
        },
        {
            "query": "Can you write a custom Python script to calculate my daily caloric intake?",
            "expected_answer": "I do not have that information/features to write code. Please contact support@fitlifeapp.com."
        }
    ]

    # 4. Generate bot outputs and run evaluations
    for i, scenario in enumerate(test_scenarios):
        query = scenario["query"]
        expected = scenario["expected_answer"]
        
        print(f"Generating bot response for query: '{query}'...")
        bot_res = get_bot_response(query)
        
        print(f"\n--- Evaluating Test Case {i+1} ---")
        print(f"Query (Inputs): {query}")
        print(f"Bot Output (Outputs): {bot_res['response']}")
        print(f"Reference Answer: {expected}")
        print(f"Retrieved Context: {bot_res['context']}\n")

        # Run Correctness Evaluator (evaluates actual output against expected output)
        print("Measuring Correctness...")
        # CORRECTNESS_PROMPT expects: inputs, outputs, and reference_outputs
        # OpenEvals dynamically maps dictionary keys or formatted strings
        correctness_res = correctness_evaluator(
            inputs=query,
            outputs=bot_res["response"],
            reference_outputs=expected
        )
        print(f"-> Correctness Score: {correctness_res['score']} (Type: {type(correctness_res['score']).__name__})")
        print(f"-> Reason: {correctness_res['comment']}")

        # Run Hallucination Evaluator (evaluates if actual output is supported by context)
        print("\nMeasuring Hallucination...")
        hallucination_res = hallucination_evaluator(
            inputs=query,
            outputs=bot_res["response"],
            context=bot_res["context"],
            reference_outputs=expected
        )
        print(f"-> Hallucination Score (True = has hallucination, False = clean): {hallucination_res['score']}")
        print(f"-> Reason: {hallucination_res['comment']}")
        print("-"*40)

if __name__ == "__main__":
    run_openevals_demo()
