import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "तुम वेदांत के विशेषज्ञ हो। नीचे दिए गए संदर्भ के आधार पर प्रश्न का उत्तर केवल हिंदी में दो:\n\n"
        "संदर्भ:\n{context}\n\n"
        "प्रश्न: {question}"
    )
)
def query_vedanta_bot():
    vectordb = FAISS.load_local(
        "vedanta_vectorstore",
        OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        allow_dangerous_deserialization = True
    )

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY),
        retriever=vectordb.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    print("🧘‍♂️ Vedanta Chatbot (type 'exit' to quit)")

    while True:
        question = input("\n🔍 Your question: ")
        if question.lower() in ["exit", "quit"]:
            break
        answer = qa.run(question)
        print(f"🧠 Answer: {answer}")

if __name__ == "__main__":
    query_vedanta_bot()
