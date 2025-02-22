import os
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.vectorstores import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import psycopg2

load_dotenv()

supabase_user = os.environ.get("user")
supabase_password = os.environ.get("password")
supabase_host = os.environ.get("host")


connection_string = f"postgresql://{supabase_user}:{supabase_password}@{supabase_host}:5432/postgres" #connection string

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def ensure_pgvector():
    try:
        conn = psycopg2.connect(connection_string)
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Failed to create vector extension: {str(e)}")
        raise e

class GeminiEmbeddings:
    def __init__(self):
        pass
    
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            try:
                embedding = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(embedding['embedding'])
            except Exception as e:
                st.error(f"Error generating embedding: {str(e)}")
                raise e
        return embeddings
    
    def embed_query(self, text):
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_query"
            )
            return response['embedding']
        except Exception as e:
            st.error(f"Error generating query embedding: {str(e)}")
            raise e

def process_document(pdf_text: str) -> VectorStore:   
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_text(pdf_text)
        documents = [Document(page_content=t) for t in texts]
        
        vector_store = PGVector.from_documents(
            documents=documents,
            embedding=GeminiEmbeddings(),
            collection_name="pdf_embeddings",
            connection_string=connection_string,
            pre_delete_collection=True
        )
        return vector_store
    except Exception as e:
        raise Exception(f"Failed to create vector store: {str(e)}")

def query_document(vector_store: VectorStore, query: str, k: int = 3):
    try:
        docs = vector_store.similarity_search(query, k=k)
        return docs
    except Exception as e:
        st.error(f"Error during similarity search: {str(e)}")
        raise e

def generate_response(context: str, query: str) -> str:
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        You are an AI assistant designed to analyze and extract information directly from uploaded PDF documents. Your primary goal is to provide accurate, concise, and contextually relevant answers strictly based on the content of the PDF. If a question cannot be answered using the provided document, clearly state that the information is not available instead of making assumptions or fabricating answers.

        If Uncertain, Respond: “The provided PDF does not contain enough information to answer that question.”
        Context: {context}
        
        Question: {query}
        
        Answer:"""
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        raise e

def main():
    st.header("Chat with Doc 💬")    
    try:
        ensure_pgvector() # Initial checks
    except Exception as e:
        st.error("Failed to initialize application requirements : pgvector()")
        return

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    pdf = st.file_uploader("Upload your Doc", type='pdf')
    if pdf is not None:
        try:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    try:
                        st.session_state.vector_store = process_document(text)
                        st.success("Document processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                        st.error(f"Error details: {type(e).__name__}")
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")

    query = st.text_input("Ask questions about your PDF file:")
    if query and st.session_state.vector_store:
        with st.spinner("Searching for answer..."):
            try:
                docs = query_document(st.session_state.vector_store, query)
                context = "\n".join([doc.page_content for doc in docs])
                response = generate_response(context, query)
                st.write("Answer:", response)
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()