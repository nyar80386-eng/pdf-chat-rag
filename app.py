import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
import os

st.set_page_config(page_title="PDF Chat", page_icon="📄")
st.title("📄 PDF Chat - Apni PDF Se Sawal Poocho")

# API Key
api_key = st.sidebar.text_input("Google Gemini API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    uploaded_file = st.file_uploader("Apni PDF upload karo", type="pdf")

    if uploaded_file is not None:
        with st.spinner("PDF process ho rahi hai..."):
            # Temp file mein save karo
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load aur chunk karo
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(pages)

            # Vector database
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

            st.session_state.vectorstore = vectorstore
            st.success(f"PDF ready hai! ({len(chunks)} chunks bane)")

    if "vectorstore" in st.session_state:
        question = st.text_input("Apna sawal likho:")

        if question:
            with st.spinner("Jawab dhoonda ja raha hai..."):
                llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
                docs = st.session_state.vectorstore.similarity_search(question, k=3)
                context = "\n\n".join([d.page_content for d in docs])

                prompt = f"""Neeche diye gaye document ke context ka use karke sawal ka jawab do.

Context:
{context}

Sawal: {question}

Jawab:"""
                response = llm.invoke(prompt)
                answer = response.content[0]['text'] if isinstance(response.content, list) else response.content

                st.write("### Jawab:")
                st.write(answer)
else:
    st.info("Pehle sidebar mein apni Gemini API key daalo")
