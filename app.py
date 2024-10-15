import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
import time

# Streamlit layout for entering API keys
api_keys = st.columns(2)
openai_api_key_col, groq_api_key_col = api_keys

# Input fields for OpenAI API key and GROQ API key
openai_api_key = openai_api_key_col.text_input("OPENAI API Key", type="password")
groq_api_key = groq_api_key_col.text_input("GROQ API Key", type="password")

# Check if both keys are entered
if not openai_api_key or not groq_api_key:
    st.error("Please enter both OpenAI and GROQ API keys to continue.")
else:
    # Proceed with the rest of the application logic

    # Streamlit title and setup
    st.title("Deblase Q&A Assistant")

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

    def summarize_with_openai(text):
        summarize_prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
        prompt_template = PromptTemplate(template=summarize_prompt, input_variables=[])
        chain = LLMChain(llm=openai_llm, prompt=prompt_template)
        summary = chain.run({"input": text})  # Pass the input text here
        return summary
    
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

            # Initialize summarizer with OpenAI model
            summarizer = load_summarize_chain(openai_llm)
            
            # Generate summaries for each document chunk
            summaries = [summarizer.run({"input_documents": [doc]}) for doc in st.session_state.final_documents]
            st.session_state.summaries = summaries  # Store summaries

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

    if prompt1:
        if "vectors" in st.session_state:
            # Retrieve similar documents
            retriever = st.session_state.vectors.as_retriever()
            relevant_docs = retriever.get_relevant_documents(prompt1)

            # Summarize retrieved documents
            context_summaries = [summarize_with_openai(doc.page_content) for doc in relevant_docs]

            # Combine summaries and retrieved documents for context
            context = "\n\n".join(context_summaries) + "\n\n" + "\n\n".join([doc.page_content for doc in relevant_docs])

            # Use OpenAI LLM to answer the question based on the context
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
