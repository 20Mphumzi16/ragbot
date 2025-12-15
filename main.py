import re
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ========== Remove DeepSeek <think> output ==========
def remove_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ========== Setup LLM ==========
model = OllamaLLM(model="mistral")

template = """
You are an expert assistant that answers questions about a graduate programme.

Use ONLY the information provided in the retrieved documents.
Do NOT invent information — reply with:
"I don’t have that information in my knowledge base."
if the answer isn’t in the documents.

Here are the relevant answers: {answers}

Here is the user's question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


# ========== Load Vector DB ==========
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Chroma(
    collection_name="grad_collection",
    persist_directory="./chrome_langchain_db",
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})


print("System is ready. Ask any question.\n")


# ========== Question loop ==========
while True:
    print("\n----------------------------------")
    question = input("Ask a question (q to quit): ")

    if question.lower() == "q":
        break

    answers = retriever.invoke(question)
    result = chain.invoke({"answers": answers, "question": question})
    print(remove_thinking(result))
