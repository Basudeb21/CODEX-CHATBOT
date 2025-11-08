import os
# Silence TensorFlow logs (1=errors only, 2=warnings+errors)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import requests
from db_insert import save_client, save_jobseeker
from sentence_transformers import SentenceTransformer
import chromadb




# Load FAQ data
with open("faq_data.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

# Initialize local ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection("faqs")

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Add data to ChromaDB (only once)
if collection.count() == 0:
    print("Indexing FAQs in ChromaDB...")
    texts = [item["label"] for item in faq_data]
    embeddings = model.encode(texts).tolist()
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=faq_data,  # store full label + answer
        ids=[str(i) for i in range(len(faq_data))]
    )


# Store user state and conversation info
user_state = {}
conversation_ended = set()
faq_counter = {}  # Track how many FAQs each user has received
completed_forms = {}  # Track which forms each user has completed


def search_faq(question, top_n=3):
    """Search FAQs using semantic similarity with ChromaDB."""
    query_vec = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_vec, n_results=top_n)
    return [
        f"Q: {meta['label']}\nA: {meta['answer']}"
        for meta in results["metadatas"][0]
    ]



def ask_codex(question, user_id="default", model="llama3.2:3b"):
    q = question.lower().strip()

    # Initialize completed forms tracker for new users
    if user_id not in completed_forms:
        completed_forms[user_id] = set()

    # --- If conversation already ended ---
    if user_id in conversation_ended:
        conversation_ended.discard(user_id)
        user_state.pop(user_id, None)
        faq_counter.pop(user_id, None)

    # --- Continue form filling if user in flow ---
    if user_id in user_state:
        # --- Handle /quit inside form only ---
        if q == "/quit":
            del user_state[user_id]
            return "👌 No problem! I've canceled the form. You can still ask FAQs or start again anytime."

        state = user_state[user_id]
        form_type = state["type"]
        quit_hint = "💡 (If you wish to quit anytime, type /quit)"

        # CLIENT FORM FLOW
        if form_type == "client":
            if "name" not in state:
                state["name"] = question
                return f"Got it 👍 Please share your email address:\n{quit_hint}"
            elif "email" not in state:
                state["email"] = question
                return f"Got it 👍 Please share your contact number:\n{quit_hint}"
            elif "phone" not in state:
                state["phone"] = question
                return f"Thanks! What tech stack are you interested in for your project?\n{quit_hint}"
            elif "tech_stack" not in state:
                state["tech_stack"] = question
                return f"Nice choice! Could you describe your project idea briefly?\n{quit_hint}"
            elif "project_description" not in state:
                state["project_description"] = question
                msg = save_client(state)
                del user_state[user_id]
                completed_forms[user_id].add("client")
                completed_forms[user_id].add("jobseeker")  # Block all forms after one submission
                
                return f"{msg}\n💼 Our team will contact you soon.\n🤖 Thank you for choosing CodeX Technolife!\n\n✨ Feel free to ask more questions or type 'exit' to end the chat."

        # JOB SEEKER FORM FLOW
        elif form_type == "jobseeker":
            if "name" not in state:
                state["name"] = question
                return f"Thanks! Please share your email address:\n{quit_hint}"
            elif "email" not in state:
                state["email"] = question
                return f"Got it. Could you share your resume link (Google Drive or any public URL)?\n{quit_hint}"
            elif "resume_link" not in state:
                state["resume_link"] = question
                return f"Perfect! Lastly, list your top skills (comma-separated):\n{quit_hint}"
            elif "skills" not in state:
                state["skills"] = question
                msg = save_jobseeker(state)
                del user_state[user_id]
                completed_forms[user_id].add("jobseeker")
                completed_forms[user_id].add("client")  # Block all forms after one submission
                
                return f"{msg}\n😊 Our HR team will review your application and reach out soon!\n🤖 Thank you for connecting with CodeX Technolife!\n\n✨ Feel free to ask more questions or type 'exit' to end the chat."

    # --- Intent detection ---
    if q.lower().strip() in ["start project", "project form", "client form","a","b"]:
        # Check if already submitted
        if "client" in completed_forms[user_id]:
            return "✅ You've already submitted a project inquiry in this session.\n💡 If you need to make changes, please contact our team directly or restart the chat."
        
        user_state[user_id] = {"type": "client"}
        return "💼 Great! Let's start your project inquiry. Please tell me your full name.\n💡 (If you wish to quit anytime, type /quit)"

    if q.lower().strip() in ["start job", "apply job", "job form"]:
        # Check if already submitted
        if "jobseeker" in completed_forms[user_id]:
            return "✅ You've already submitted a job application in this session.\n💡 If you need to make changes, please contact our HR team directly or restart the chat."
        
        user_state[user_id] = {"type": "jobseeker"}
        return "👩‍💻 Awesome! Let's start your application. What's your full name?\n💡 (If you wish to quit anytime, type /quit)"

    # --- FAQ Handling ---
    context_docs = search_faq(question)
    if not context_docs:
        return "⚠️ I can only answer questions related to Codex Technolife Pvt. Ltd. Please ask something about our company or services."

    context = "\n".join(context_docs)
    prompt = f"""
    You are the official chatbot of CodeX Technolife Pvt. Ltd.
    Answer the user's question clearly, naturally, and professionally.

    Use only the information provided below to answer. 
    If the answer is not available in the information, do NOT assume, guess, or make up any details.
    Instead, politely guide the user to contact our team by saying:
    "You can reach us directly by phone at +91 947 456 2952 or email us at info@codextechnolife.com."

    Never say phrases like "I couldn't find that," "the context doesn't mention," or "according to the information."
    Speak confidently as a company representative who has this knowledge.

    Information:
    {context}

    User Question:
    {question}

    Provide a direct, confident answer in 2–4 lines as if you are the company's assistant.
    """


    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "max_tokens": 400},
            timeout=120
        )

        lines = res.text.strip().split("\n")
        answer = ""
        for line in lines:
            try:
                obj = json.loads(line)
                answer += obj.get("response", "")
            except:
                continue

        answer = answer.strip()

        # --- Track FAQ responses ---
        faq_counter[user_id] = faq_counter.get(user_id, 0) + 1

        # After 3–4 FAQ answers, offer forms not yet completed
        if faq_counter[user_id] >= 3:
            faq_counter[user_id] = 0  # reset counter
            
            suggestions = []
            if "client" not in completed_forms[user_id]:
                suggestions.append("discuss a project - type: \"A\"")
            if "jobseeker" not in completed_forms[user_id]:
                suggestions.append("apply for a job - type: \"B\"")
            
            if suggestions:
                suggestion_text = " or ".join(suggestions)
                return answer + f"\n\n💡 By the way, to {suggestion_text}, or you can continue asking questions!"

        return answer
    except Exception as e:
        return f"⚠️ AI error: {e}"


# --- MAIN LOOP ---
if __name__ == "__main__":
    print("🤖 Codex FAQ Chatbot Ready! Type 'exit' to quit the chat.\n")
    user_id = "default"

    while True:
        user_input = input("You: ")

        # End entire chat
        if user_input.lower() in ["exit"]:
            print("🤖 It was great chatting with you! 😊")
            print("💡 Have a wonderful day ahead — and don't hesitate to return if you need help again. 👋")
            break

        response = ask_codex(user_input, user_id)
        print("Bot:", response)
        print("-" * 50)