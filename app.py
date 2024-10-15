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
api_keys = st.columns(1)
openai_api_key_col = api_keys

# groq_api_key = groq_api_key_col.text_input("GROQ API Key", type="password")
openai_api_key = openai_api_key_col.text_input("OPENAI API Key", type="password")

# Check if both keys are entered
if not openai_api_key:
    st.error("Please enter OpenAI API keys to continue.", icon="❗️")
else:
    # Streamlit title and setup
    st.title("Document Q&A with OpenAI Assistant")
    
    # Initialize the OpenAI GPT model for Q&A
    openai_llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
    
    # Define the prompt template for OpenAI assistant to answer based on context
    prompt_template = """
    You are an assistant helping with document retrieval. Based on the following context, answer the question:
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    
    # Function to handle document embedding
    def vector_embedding():
        if "vectors" not in st.session_state:
            # Initialize OpenAI embeddings with the loaded API key
            st.session_state.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            
            # Load documents from the directory
            st.session_state.loader = PyPDFDirectoryLoader("./debase_doc_c")  # Data Ingestion
            st.session_state.docs = st.session_state.loader.load()  # Document Loading
            
            # Ensure documents are not empty
            if len(st.session_state.docs) == 0:
                st.error("No documents found. Please ensure the documents are properly loaded.")
                return
            
            # Split documents into chunks for better embedding
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
    
            # Ensure the documents were split properly
            if len(st.session_state.final_documents) == 0:
                st.error("No document chunks were created. Please check the document loading and splitting process.")
                return
    
            # Generate embeddings for the document chunks
            try:
                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
                st.success("Document embedding and vector store created successfully!")
            except IndexError as e:
                st.error(f"Failed to create vector store: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred during embedding: {str(e)}")
    
    # Input for asking questions from documents
    prompt1 = st.text_input("Enter Your Question From Documents")
    
    # Button to trigger document embedding
    if st.button("Documents Embedding"):
        vector_embedding()
        st.write("Vector Store DB Is Ready")
    
    # Handle question-answering if a question is asked
    if prompt1:
        if "vectors" in st.session_state:
            # Retrieve similar documents based on the user's query
            retriever = st.session_state.vectors.as_retriever()
            relevant_docs = retriever.get_relevant_documents(prompt1)
            
            # Combine the retrieved documents into a single context string
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
            # Use OpenAI's LLM (GPT) to answer the question based on the context
            chain = LLMChain(llm=openai_llm, prompt=prompt)
            response = chain.run({"context": context, "question": prompt1})
    
            # Display the response
            st.write(response)
    
            # Optionally, display the relevant document chunks in an expander
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(relevant_docs):
                    st.write(f"Document {i + 1}:")
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        else:
            st.error("Please embed the documents first by clicking 'Documents Embedding'.")


    
           
