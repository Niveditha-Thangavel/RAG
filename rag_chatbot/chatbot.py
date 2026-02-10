from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from pypdf import PdfReader
import faiss
import numpy as np
import os, json, dotenv
import ollama
import logging

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

logging.getLogger("pypdf").setLevel(logging.ERROR)

# ===============================
# ENV CONFIG (QUIET MODE)
# ===============================
os.environ["CREWAI_TRACING"] = "false"
os.environ["CREWAI_TELEMETRY"] = "false"
os.environ["CREWAI_LOG_LEVEL"] = "ERROR"
os.environ["LITELLM_LOG"] = "none"

# ===============================
# PATH SETUP (IMPORTANT)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_DIR = os.path.join(BASE_DIR, "pdfs")
MEM_FILE = os.path.join(BASE_DIR, "msg.json")
EMB_FILE = os.path.join(BASE_DIR, "embeddings.npy")
IDX_FILE = os.path.join(BASE_DIR, "faiss.index")

# ===============================
# LLM CONFIG (GEMINI)
# ===============================
llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=GEMINI_API_KEY
)

# ===============================
# LOAD PDFs
# ===============================
documents = []

for file in os.listdir(PDF_DIR):
    if file.endswith(".pdf"):
        reader = PdfReader(os.path.join(PDF_DIR, file))
        for page in reader.pages:
            text = page.extract_text()
            if text:
                documents.append(text)

print(f"📄 Loaded {len(documents)} pages from PDFs")

# ===============================
# CHUNKING
# ===============================
def chunk_text(text, max_len=900):
    paragraphs = [
        p.strip()
        for p in text.split("\n")
        if len(p.strip()) > 30
    ]

    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) <= max_len:
            current += " " + p
        else:
            chunks.append(current.strip())
            current = p

    if current:
        chunks.append(current.strip())

    return chunks

chunks = []
for doc in documents:
    chunks.extend(chunk_text(doc))

print(f"🧩 Total semantic chunks: {len(chunks)}")

# ===============================
# EMBEDDINGS + FAISS (CACHED)
# ===============================
def embed_texts(texts):
    vectors = []
    for t in texts:
        res = ollama.embeddings(
            model="nomic-embed-text",
            prompt=t
        )
        vectors.append(res["embedding"])
    return np.array(vectors).astype("float32")

if os.path.exists(EMB_FILE) and os.path.exists(IDX_FILE):
    print("🔁 Loading cached embeddings & FAISS index...")
    embeddings = np.load(EMB_FILE)
    index = faiss.read_index(IDX_FILE)
else:
    print("🧠 Creating embeddings (one-time process)...")
    embeddings = embed_texts(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    np.save(EMB_FILE, embeddings)
    faiss.write_index(index, IDX_FILE)

    print("✅ Embeddings & FAISS index saved")

# ===============================
# RAG TOOL
# ===============================
class RagTool(BaseTool):
    name: str = "rag_tool"
    description: str = "Retrieve factual evidence from the PDF knowledge base"

    def _run(self, query: str) -> str:
        q_emb = embed_texts([query])
        _, idxs = index.search(q_emb, k=5)

        evidence = []
        for i in idxs[0]:
            evidence.append(f"[EVIDENCE]\n{chunks[i]}")

        return "\n\n".join(evidence)

# ===============================
# AGENT
# ===============================
def create_agent():
    return Agent(
        role="Friendly Knowledge Assistant",
        goal="Answer using document evidence with conversational awareness",
        backstory=(
            "You are Lara, a friendly and natural conversational assistant.\n"
            "You are given conversation context and document evidence.\n\n"
            "RULES:\n"
            "- Be warm and conversational\n"
            "- Use conversation context naturally if relevant\n"
            "- ALL factual answers MUST come from document evidence\n"
            "- If the documents do not contain the answer, politely say so\n"
            "- Never invent facts\n"
            "- Do not mention tools or internal reasoning\n"
            "Output ONLY the final answer\n"
        ),
        tools=[RagTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=1,
        reasoning=False
    )

# ===============================
# FORMAT CHAT HISTORY (MEMORY)
# ===============================
def format_history(history, max_turns=4):
    recent = history[-max_turns * 2:]
    lines = []
    for h in recent:
        role = "User" if h["role"] == "user" else "Assistant"
        lines.append(f"{role}: {h['text']}")
    return "\n".join(lines)

# ===============================
# CHAT EXECUTION
# ===============================
def run_chat(msg, history):
    agent = create_agent()

    conversation_context = format_history(history)

    task = Task(
        description=(
            "You are in an ongoing conversation with the user.\n\n"
            "Conversation so far:\n"
            f"{conversation_context}\n\n"
            f"User message:\n{msg}\n\n"
            "INSTRUCTIONS:\n"
            "- Respond naturally and conversationally\n"
            "- Maintain context from earlier messages if relevant\n"
            "- There are TWO types of information:\n"
            "1. Conversational context (previous messages, user name)\n"
            "2. Document knowledge (PDF content)\n"
            "- Use conversational context ONLY for personal or conversational questions\n"
            "(e.g., names, greetings, continuity)\n"
            "- Use rag_tool ONLY for factual or document-related questions\n"
            "- ALL factual/document answers MUST come from document evidence\n"
            "- Do NOT say you cannot remember personal details if they appear in the conversation history"
            "- If the documents do not contain the answer, say so politely\n"
        ),
        expected_output="Friendly, context-aware, document-grounded answer",
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )

    return crew.kickoff()

# ===============================
# CHAT LOOP (WITH MEMORY)
# ===============================
if __name__ == "__main__":

    if os.path.exists(MEM_FILE):
        with open(MEM_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []

    print("🤖 Lara – Friendly RAG PDF Chatbot Ready")
    print("(type 'bye' to exit)\n")

    while True:
        msg = input("You: ").strip()

        if msg.lower() == "bye":
            print("Lara: Bye 😊")
            break

        result = run_chat(msg, history)

        answer = (
            result.final_output
            if hasattr(result, "final_output")
            else str(result)
        )

        print("Lara:", answer)

        history.append({"role": "user", "text": msg})
        history.append({"role": "assistant", "text": answer})

        with open(MEM_FILE, "w") as f:
            json.dump(history, f, indent=2)