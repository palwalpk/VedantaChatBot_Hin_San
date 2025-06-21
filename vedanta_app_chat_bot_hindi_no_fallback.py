import streamlit as st
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# Set your OpenAI API key
openai_key = st.secrets["OPENAI_API_KEY"]

# Hindi prompt template to avoid fallback
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    नीचे दिए गए संदर्भ का उपयोग करके केवल हिंदी में उत्तर दें। अगर उत्तर संदर्भ में नहीं है तो कहें "माफ़ कीजिए, मुझे इस विषय में जानकारी नहीं है।"

    संदर्भ: {context}
    प्रश्न: {question}
    उत्तर:
    """
)

# Load Hindi documents
@st.cache_resource
def load_vector_store():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectordb = FAISS.load_local(
        "vedanta_vectorstore",
        embeddings,
        allow_dangerous_deserialization=True)
    return vectordb

vectordb = load_vector_store()

# Chat model (set to Hindi)
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Custom QA chain with fallback prevention
def get_hindi_answer(query):
    docs = vectordb.similarity_search(query, k=3)
    if not docs:
        return "माफ़ कीजिए, मुझे इस विषय में जानकारी नहीं है।"

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = custom_prompt.format(context=context, question=query)
    response = llm.predict(prompt)
    return response

# Streamlit UI
st.title("🕉️ वेदांत हिंदी चैटबॉट")

user_question = st.text_input("प्रश्न पूछें:")

if user_question:
    with st.spinner("उत्तर खोजा जा रहा है..."):
        answer = get_hindi_answer(user_question)
        st.write("**उत्तर:**")
        st.success(answer)
