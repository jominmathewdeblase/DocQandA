import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings  # OpenAI embeddings
# from dotenv import load_dotenv
import time



# Layout for API keys input
api_keys = st.columns(2)
groq_api_key_col, openai_api_key_col = api_keys

groq_api_key = groq_api_key_col.text_input("GROQ API Key", type="password")
openai_api_key = openai_api_key_col.text_input("OPENAI API Key", type="password")

# Check if both keys are entered
if not groq_api_key or not openai_api_key:
    st.error("Please enter both GROQ and Google API keys to continue.", icon="❗️")
else:


    # Streamlit title and setup
    st.title("Deblase Document Q&A")
    
    # Initialize the language model (GROQ model for Q&A)
    groq_api_key = os.getenv('GROQ_API_KEY')
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-groq-70b-8192-tool-use-preview")
    
    # Define the prompt template for Q&A
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        <context>
        Questions:{input}
        """
    )
    
    # Function for vector embedding using OpenAI embeddings
    def vector_embedding():
        if "vectors" not in st.session_state:
            # Initialize OpenAI embeddings with the loaded API key
            st.session_state.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            
            # Load documents from the directory
            st.session_state.loader = PyPDFDirectoryLoader("./debase_doc_c")  # Data Ingestion
            st.session_state.docs = st.session_state.loader.load()  # Document Loading
            
            # Split documents into chunks for better embedding
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
    
            # Check if there are any documents to create the index
            if len(st.session_state.final_documents) > 0:
                # Use FAISS to create a vector store with OpenAI embeddings
                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
                st.success("Document embedding and vector store created successfully!")
            else:
                st.warning("No documents loaded or split. Please check data loading and splitting.")
    
    # Input for asking questions from documents
    prompt1 = st.text_input("Enter Your Question From Documents")
    
    # Button to trigger document embedding
    if st.button("Documents Embedding"):
        vector_embedding()
        st.write("Vector Store DB Is Ready")
    
    # Handle question-answering if a question is asked
    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Ensure that the vector store has been created
        if "vectors" in st.session_state:
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Measure response time
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write("Response time: ", time.process_time() - start)
            
            # Display the answer
            st.write(response['answer'])
    
            # With a streamlit expander, show relevant document chunks
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response.get("context", [])):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        else:
            st.error("Please embed the documents first by clicking 'Documents Embedding'.")


