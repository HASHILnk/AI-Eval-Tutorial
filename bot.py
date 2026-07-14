import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Check for API key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable not found in .env file.")

# Define the knowledge base of our FitLife App QA bot
KNOWLEDGE_BASE = {
    "subscription": (
        "FitLife offers two subscription tiers:\n"
        "1. Free Tier: Basic workouts, ad-supported, no custom diet plans.\n"
        "2. Premium Tier: $9.99/month, includes unlimited workouts, personalized diet plans, "
        "offline access, and ad-free experience. Subscriptions can be canceled at any time in the app settings."
    ),
    "features": (
        "FitLife features include:\n"
        "1. Workout Tracker: Log daily exercises, sets, and reps.\n"
        "2. Diet & Nutrition Planner: Track calories, macros, and get personalized meal plans (Premium only).\n"
        "3. Live Classes: Daily interactive streaming sessions with certified trainers."
    ),
    "refund": (
        "Refund Policy: FitLife subscriptions are refundable within 14 days of purchase. "
        "To request a refund, contact support@fitlifeapp.com with your invoice ID."
    )
}

def retrieve_context(query: str) -> str:
    """
    Simple mock retriever that finds the most relevant knowledge snippet based on keywords.
    """
    query_lower = query.lower()
    retrieved = []
    
    if "sub" in query_lower or "membership" in query_lower or "cost" in query_lower or "price" in query_lower or "tier" in query_lower:
        retrieved.append(KNOWLEDGE_BASE["subscription"])
    if "feature" in query_lower or "track" in query_lower or "diet" in query_lower or "meal" in query_lower or "class" in query_lower:
        retrieved.append(KNOWLEDGE_BASE["features"])
    if "refund" in query_lower or "money back" in query_lower or "return" in query_lower:
        retrieved.append(KNOWLEDGE_BASE["refund"])
        
    if not retrieved:
        # Default fallback context
        return "FitLife is a fitness tracking and nutrition coaching app that helps you reach your wellness goals."
        
    return "\n\n".join(retrieved)

def get_bot_response(query: str) -> dict:
    """
    Queries the Groq LLM (llama-3.3-70b-versatile) using retrieved context.
    Returns a dictionary with the 'response' and the 'retrieved_context'.
    """
    context = retrieve_context(query)
    
    client = Groq()
    
    system_prompt = (
        "You are FitLifeBot, a friendly and helpful customer support assistant for the FitLife app. "
        "Answer the user's question accurately using ONLY the provided context. Do NOT make up information "
        "that is not explicitly mentioned in the context. If you do not know the answer, say "
        "'I am sorry, but I do not have that information. Please contact support@fitlifeapp.com.'\n\n"
        f"Context:\n{context}"
    )
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.0
    )
    
    response_text = completion.choices[0].message.content
    
    return {
        "query": query,
        "response": response_text,
        "context": context
    }

if __name__ == "__main__":
    # Test run
    test_query = "How much does the premium subscription cost, and what do I get with it?"
    result = get_bot_response(test_query)
    print(f"Query: {result['query']}\n")
    print(f"Context:\n{result['context']}\n")
    print(f"Response:\n{result['response']}\n")
