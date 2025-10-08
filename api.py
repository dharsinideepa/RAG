from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import os
import uvicorn
import logging
from typing import List, Dict, Any, Optional

# Import our simplified RAG implementation
from simple_rag import SimpleLegalRAG

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Legal Support RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the Together API key from environment variable
together_api_key = os.environ.get("TOGETHER_API_KEY", "f0bede9bb2af9368add296406997dc4fc7643a6aa3acf2b91ae49b005b14953c")

# Initialize the RAG system with Together API
legal_rag = SimpleLegalRAG(together_api_key=together_api_key)

# Setup the system
legal_rag.setup()

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, str]]

class DocumentResponse(BaseModel):
    filename: str
    status: str
    message: str

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    """
    Serve the frontend HTML directly from the API.
    """
    try:
        with open("index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error serving HTML: {str(e)}")
        return HTMLResponse(content="<h1>Error loading page</h1>")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a legal question.
    """
    try:
        result = legal_rag.query(request.question)
        return result
    except Exception as e:
        logger.error(f"Error querying RAG system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to the RAG system.
    """
    try:
        # Check file extension
        if not file.filename.endswith(('.pdf', '.txt', '.csv')):
            return JSONResponse(
                status_code=400,
                content={
                    "filename": file.filename,
                    "status": "error",
                    "message": "Unsupported file type. Please upload PDF, TXT, or CSV files."
                }
            )
        
        # Save the uploaded file
        file_location = os.path.join(legal_rag.docs_dir, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Add the document to the RAG system
        legal_rag.add_document(file_location)
        
        return {
            "filename": file.filename,
            "status": "success",
            "message": "Document uploaded and processed successfully"
        }
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_system():
    """
    Reset the RAG system.
    """
    try:
        legal_rag.reset()
        return {"status": "success", "message": "System reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[str])
async def list_documents():
    """
    List all documents in the system.
    """
    try:
        documents = []
        for root, _, files in os.walk(legal_rag.docs_dir):
            for file in files:
                if file.endswith(('.pdf', '.txt', '.csv')):
                    rel_path = os.path.relpath(os.path.join(root, file), legal_rag.docs_dir)
                    documents.append(rel_path)
        logger.info(f"Found {len(documents)} documents: {documents}")
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a document from the system.
    """
    try:
        success = legal_rag.remove_document(filename)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found or could not be removed")
        
        return {
            "filename": filename,
            "status": "success",
            "message": "Document deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug")
async def debug_system():
    """
    Debug endpoint to check system status.
    """
    try:
        # Check physical documents
        physical_docs = []
        for root, _, files in os.walk(legal_rag.docs_dir):
            for file in files:
                if file.endswith(('.pdf', '.txt', '.csv')):
                    physical_docs.append(os.path.join(root, file))
        
        # Check vector database
        vectordb_status = "Not initialized"
        vectordb_count = 0
        if legal_rag.vectordb:
            vectordb_status = "Initialized"
            try:
                vectordb_count = legal_rag.vectordb._collection.count()
            except Exception as e:
                vectordb_status = f"Error: {str(e)}"
        
        # Check directory permissions
        docs_dir_writable = os.access(legal_rag.docs_dir, os.W_OK)
        persist_dir_writable = os.access(legal_rag.persist_dir, os.W_OK)
        
        return {
            "physical_documents": physical_docs,
            "physical_document_count": len(physical_docs),
            "documents_directory": legal_rag.docs_dir,
            "vectordb_directory": legal_rag.persist_dir,
            "vectordb_status": vectordb_status,
            "vectordb_count": vectordb_count,
            "docs_dir_writable": docs_dir_writable,
            "persist_dir_writable": persist_dir_writable,
            "qa_chain_initialized": legal_rag.qa_chain is not None,
            "llm_initialized": legal_rag.llm is not None,
            "embeddings_initialized": legal_rag.embeddings is not None
        }
    except Exception as e:
        logger.error(f"Debug error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)