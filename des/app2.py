import os
import time
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from htmltemplates import css,bot_template,user_template

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A Chatbot")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context .
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(raw_text)
   # embeddings = Ollam
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    return vectors

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response=st.session_state.conversation({'question':user_question})
    st.session_state.chat_history=response['chat_history']

    for i,message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
            

def main():
    
    
    st.write(css,unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None
    


    user_question = st.text_input("Enter Your Question From Documents")


    if user_question:
        handle_userinput(user_question)
    
   

    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_files = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type=['pdf'])
        
        if st.button("Process PDFs"):
            if pdf_files:
                with st.spinner("Processing"):
                    st.session_state.vectors = vector_embedding(pdf_files)
                    st.session_state.conversation = get_conversation_chain(st.session_state.vectors)
                    st.success("Processing complete!")
            else:
                st.error("Please upload PDFs first.")
    
    if st.button("Ask"):
        with st.spinner("Processing"):
            if "vectors" in st.session_state:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
                start_time = time.process_time()
                response = retrieval_chain.invoke({'input': user_question})
                end_time = time.process_time()
            
                st.write("Response time:", end_time - start_time)
               # st.write(response['answer'])
            
                with st.expander("Document Similarity Search"):
                    for i, doc in enumerate(response["context"]):
                        st.write(doc.page_content)
                        st.write("--------------------------------")
            else:
                st.error("No documents processed yet. Please upload and process PDFs first.")

if __name__ == '__main__':
    main()
