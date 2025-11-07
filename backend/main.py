import os
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form, HTTPException
from pydantic import BaseModel

# Import our custom processing functions
from processing import process_user_files, query_user_data, UPLOAD_DIR

app = FastAPI(
    title="Multimodal RAG API",
    description="Upload PDFs and query them using Multimodal RAG with Gemini and ChromaDB."
)

# Helper Function
def save_upload_file(upload_file: UploadFile, destination: Path):
    """Saves an uploaded file to a destination."""
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

# Pydantic Models (for request/response validation)
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

# --- API Endpoints ---
@app.post("/upload/", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Uploads a PDF file for a specific user.
    Ingestion is handled as a background task.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        
    user_upload_dir = UPLOAD_DIR / user_id
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_destination = user_upload_dir / file.filename
    save_upload_file(file, file_destination)
    
    # Add the processing job to the background
    # This allows the API to respond immediately
    background_tasks.add_task(process_user_files, user_id=user_id)
    
    return UploadResponse(
        message="File uploaded successfully. Processing started in the background.",
        user_id=user_id,
        filename=file.filename
    )

@app.post("/query/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Queries the uploaded data for a specific user.
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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multimodal RAG API. Go to /docs for API."}