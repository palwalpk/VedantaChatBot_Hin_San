import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
#URL used in Steamlit clound
#https://vedantachatbothindi.streamlit.app/
# Load OpenAI key from secrets
openai_key = st.secrets["OPENAI_API_KEY"]

# Set up Streamlit
st.set_page_config(page_title="Vedanta GPT", page_icon="🧘‍♂️")
st.title("🧘‍♂️ वेदांत GPT (Vedanta Chatbot)")
st.write("हिंदी में अपने प्रश्न पूछें और वेदांत शास्त्र से उत्तर प्राप्त करें।")

# Load Vector Store
@st.cache_resource
def load_chain():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectordb = FAISS.load_local(
        "vedanta_vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "तुम वेदांत के विशेषज्ञ हो। नीचे दिए गए संदर्भ के आधार पर प्रश्न का उत्तर केवल हिंदी में दो:\n\n"
            "संदर्भ:\n{context}\n\n"
            "प्रश्न: {question}"
        )
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", openai_api_key=openai_key),
        retriever=vectordb.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

qa_chain = load_chain()

# Chat UI
query = st.text_input("🔍 प्रश्न दर्ज करें:")
#if st.button("उत्तर दें") and query:
if query:
    with st.spinner("सोच रहे हैं..."):
        result = qa_chain.run(query)
        st.markdown("### 🧠 उत्तर:")
        st.write(result)