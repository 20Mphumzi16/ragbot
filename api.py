from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Remove DeepSeek `<think>` output ==========
def remove_thinking(text: str) -> str:
    return re.sub(r"`<think>`.*?`</think>`", "", text, flags=re.DOTALL).strip()


# ========== Initialize components ==========
print("Initializing vector database...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = Chroma(
    collection_name="grad_collection",
    persist_directory="./chrome_langchain_db",
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

print("Initializing LLM...")
model = OllamaLLM(model="mistral")
template = """
You are an expert assistant that answers questions about a graduate programme.

Use ONLY the information provided in the retrieved documents.
Do NOT invent information — reply with:
"I don't have that information in my knowledge base."
if the answer isn't in the documents.

Here are the relevant answers: {answers}

Here is the user's question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

print("API is ready!")


# ========== Request/Response Models ==========
class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    question: str


# ========== API Endpoints ==========
@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running", "status": "ready"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question to the chatbot.
    
    - **question**: The user's question
    """
    try:
        # Retrieve relevant documents
        answers = retriever.invoke(request.question)
        
        # Generate response
        result = chain.invoke({"answers": answers, "question": request.question})
        answer = remove_thinking(result)
        
        return QuestionResponse(question=request.question, answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/chat")
async def chat(request: QuestionRequest):
    """
    Simplified chat endpoint that returns just the answer.
    """
    try:
        answers = retriever.invoke(request.question)
        result = chain.invoke({"answers": answers, "question": request.question})
        answer = remove_thinking(result)
        
        return {"answer": answer}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")