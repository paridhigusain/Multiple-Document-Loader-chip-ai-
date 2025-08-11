import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
import os
import io

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """Extract text from PDF files"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_excel_text(excel_docs):
    """Extract text from Excel files"""
    text = ""
    for excel_file in excel_docs:
        try:
            # Read Excel file - try different engines for compatibility
            try:
                # Try openpyxl first (for .xlsx files)
                df = pd.read_excel(excel_file, engine='openpyxl')
            except:
                # Fallback to xlrd (for .xls files)
                df = pd.read_excel(excel_file, engine='xlrd')
            
            # Convert DataFrame to text
            # Option 1: Include headers and data in a structured format
            text += f"\n--- Excel File: {excel_file.name} ---\n"
            text += f"Columns: {', '.join(df.columns.tolist())}\n"
            text += f"Number of rows: {len(df)}\n\n"
            
            # Convert each row to text
            for index, row in df.iterrows():
                row_text = f"Row {index + 1}: "
                for col in df.columns:
                    row_text += f"{col}: {str(row[col])}, "
                text += row_text.rstrip(", ") + "\n"
            
            text += "\n"
            
        except Exception as e:
            st.error(f"Error reading Excel file {excel_file.name}: {str(e)}")
            continue
    
    return text

def get_text_chunks(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create vector store from text chunks"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Create conversational chain"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the context contains data from Excel files, you can reference specific rows, columns, or data points.
    If the answer is not in the provided context, just say "answer is not available in the context".
    Don't provide wrong answers.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Process user input and generate response"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load vector store
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat with Multiple Documents (PDF & Excel)")
    st.header("Chat with Chip.ai ")

    user_question = st.text_input("Ask a Question from the uploaded documents")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Document Upload")
        
        # PDF upload section
        st.subheader("PDF Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files", 
            accept_multiple_files=True, 
            type=['pdf'],
            key="pdf_uploader"
        )
        
        # Excel upload section
        st.subheader("Excel Documents")
        excel_docs = st.file_uploader(
            "Upload your Excel Files", 
            accept_multiple_files=True, 
            type=['xlsx', 'xls'],
            key="excel_uploader"
        )
        
        if st.button("Submit & Process"):
            if not pdf_docs and not excel_docs:
                st.error("Please upload at least one document (PDF or Excel)")
                return
                
            with st.spinner("Processing documents..."):
                # Initialize text variable
                raw_text = ""
                
                # Process PDF files
                if pdf_docs:
                    st.info(f"Processing {len(pdf_docs)} PDF file(s)...")
                    raw_text += get_pdf_text(pdf_docs)
                
                # Process Excel files
                if excel_docs:
                    st.info(f"Processing {len(excel_docs)} Excel file(s)...")
                    raw_text += get_excel_text(excel_docs)
                
                if raw_text:
                    # Create text chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Create vector store
                    get_vector_store(text_chunks)
                    
                    st.success("Documents processed successfully! You can now ask questions.")
                else:
                    st.error("No text could be extracted from the uploaded documents.")

if __name__ == "__main__":
    main()