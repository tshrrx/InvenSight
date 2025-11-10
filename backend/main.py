import os
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# Import our custom processing functions
from processing import (
    process_user_files, 
    query_user_data, 
    list_user_files, 
    delete_user_file,
    TEMP_DIR
)

app = FastAPI(
    title="InvenSight Multimodal RAG API",
    description="Upload PDFs to S3, store embeddings in Pinecone, and query using Multimodal RAG with Gemini.",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper Function
def save_upload_file_temp(upload_file: UploadFile) -> Path:
    """Saves an uploaded file to temp directory."""
    temp_path = TEMP_DIR / upload_file.filename
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return temp_path
    finally:
        upload_file.file.close()

# Pydantic Models
class QueryRequest(BaseModel):
    user_id: str
    query: str
    
class UploadResponse(BaseModel):
    message: str
    user_id: str
    filename: str
    
class QueryResponse(BaseModel):
    user_id: str
    query: str
    answer: str

class HealthResponse(BaseModel):
    status: str
    message: str

class FileInfo(BaseModel):
    file_name: str
    s3_key: str
    size: int
    last_modified: str

# --- API Endpoints ---

@app.get("/", response_model=HealthResponse)
def read_root():
    """Root endpoint with API information."""
    return HealthResponse(
        status="healthy",
        message="Welcome to InvenSight Multimodal RAG API with Pinecone + S3. Go to /docs for API documentation."
    )

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for monitoring."""
    return HealthResponse(
        status="healthy",
        message="Service is running with Pinecone and S3"
    )

@app.post("/upload/", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Uploads a PDF file for a specific user.
    Stores PDF in S3 and embeddings in Pinecone.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    # Save temporarily
    temp_file = save_upload_file_temp(file)
    
    # Process in background
    background_tasks.add_task(process_user_files, user_id=user_id, pdf_path=temp_file)
    
    # Clean up temp file after processing
    background_tasks.add_task(temp_file.unlink, missing_ok=True)
    
    return UploadResponse(
        message="File uploaded successfully. Processing started in the background.",
        user_id=user_id,
        filename=file.filename
    )

@app.post("/query/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Queries the uploaded data for a specific user from Pinecone.
    """
    try:
        answer = query_user_data(user_id=request.user_id, query=request.query)
        
        return QueryResponse(
            user_id=request.user_id,
            query=request.query,
            answer=answer
        )
    except Exception as e:
        print(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.get("/users/{user_id}/files", response_model=List[FileInfo])
async def get_user_files(user_id: str):
    """Lists all files uploaded by a user."""
    try:
        files = list_user_files(user_id)
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {e}")

@app.delete("/users/{user_id}/files/{file_name}")
async def delete_file(user_id: str, file_name: str):
    """Deletes a specific file and its embeddings."""
    try:
        success = delete_user_file(user_id, file_name)
        if success:
            return {"message": f"Successfully deleted {file_name}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")