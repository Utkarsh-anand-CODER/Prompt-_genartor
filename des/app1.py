import os
import time
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Chat with pdf")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Questions:{input}
    """
)

def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def vector_embedding(pdfs):
    raw_text = extract_text_from_pdfs(pdfs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len,
)
    chunks = text_splitter.split_text(raw_text)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    return vectors

def main():
    user_question = st.text_input("Enter Your Question From Documents")
    
    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_files = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type=['pdf'])
        
        if st.button("Process PDFs"):
            if pdf_files:
                with st.spinner("Processing"):
                    st.session_state.vectors = vector_embedding(pdf_files)
                    st.success("Processing complete!")
            else:
                st.error("Please upload PDFs first.")
    
    if st.button("Search"):
        with st.spinner("Processing"):

            if "vectors" in st.session_state:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
                start_time = time.process_time()
                response = retrieval_chain.invoke({'input': user_question})
                end_time = time.process_time()
            
                st.write("Response time:", end_time - start_time)
                st.write(response['answer'])
            
                with st.expander("Document Similarity Search"):
                    for i, doc in enumerate(response["context"]):
                        st.write(doc.page_content)
                        st.write("--------------------------------")
            else:
                st.error("No documents processed yet. Please upload and process PDFs first.")

if __name__ == '__main__':
    main()
