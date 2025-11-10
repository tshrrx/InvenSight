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
    TEMP_DIR,
    text_index,
    image_index
)

app = FastAPI(
    title="InvenSight Multimodal RAG API",
    description="Upload PDFs to S3, store embeddings in Pinecone, and query using Multimodal RAG with Gemini.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://invensight.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processing status
processing_status: Dict[str, str] = {}

def save_upload_file_temp(upload_file: UploadFile) -> Path:
    temp_path = TEMP_DIR / upload_file.filename
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return temp_path
    finally:
        upload_file.file.close()

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

class ProcessingStatus(BaseModel):
    status: str
    message: str

@app.get("/", response_model=HealthResponse)
def read_root():
    return HealthResponse(
        status="healthy",
        message="Welcome to InvenSight Multimodal RAG API with Pinecone + S3. Go to /docs for API documentation."
    )

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="healthy", message="Service appears healthy")

@app.post("/upload/", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    temp_file = save_upload_file_temp(file)

    status_key = f"{user_id}_{file.filename}"
    processing_status[status_key] = "processing"

    def process_with_status():
        try:
            process_user_files(user_id=user_id, pdf_path=temp_file)
            processing_status[status_key] = "completed"
            print(f"Processing completed for {status_key}")
        except Exception as e:
            processing_status[status_key] = f"failed: {str(e)}"
            print(f"Processing failed for {status_key}: {e}")
        finally:
            try:
                temp_file.unlink(missing_ok=True)
            except Exception as e:
                print(f"Error deleting temp file: {e}")

    background_tasks.add_task(process_with_status)

    return UploadResponse(
        message="File uploaded; processing started.",
        user_id=user_id,
        filename=file.filename
    )

@app.get("/upload/status/{user_id}/{filename}", response_model=ProcessingStatus)
async def check_upload_status(user_id: str, filename: str):
    status_key = f"{user_id}_{filename}"
    status = processing_status.get(status_key, "not_found")
    if status == "not_found":
        message = "No processing record found for this file"
    elif status == "processing":
        message = "File is currently being processed"
    elif status == "completed":
        message = "File processing completed successfully"
    elif status.startswith("failed"):
        message = status
    else:
        message = status
    return ProcessingStatus(status=status, message=message)

@app.post("/query/", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        answer = query_user_data(user_id=request.user_id, query=request.query)
        return QueryResponse(user_id=request.user_id, query=request.query, answer=answer)
    except Exception as e:
        print(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.get("/users/{user_id}/files", response_model=List[FileInfo])
async def get_user_files(user_id: str):
    try:
        files = list_user_files(user_id)
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {e}")

@app.delete("/users/{user_id}/files/{file_name}")
async def delete_file(user_id: str, file_name: str):
    try:
        success = delete_user_file(user_id, file_name)
        if success:
            status_key = f"{user_id}_{file_name}"
            processing_status.pop(status_key, None)
            return {"message": f"Successfully deleted {file_name}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

@app.get("/debug/user/{user_id}")
async def debug_user_data(user_id: str):
    try:
        text_stats = text_index.describe_index_stats()
        image_stats = image_index.describe_index_stats()

        return {
            "user_id": user_id,
            "text_index": {
                "total_vectors": text_stats.total_vector_count,
                "user_namespace_exists": user_id in text_stats.namespaces if text_stats.namespaces else False,
                "user_vector_count": text_stats.namespaces.get(user_id, {}).get('vector_count', 0) if text_stats.namespaces and user_id in text_stats.namespaces else 0
            },
            "image_index": {
                "total_vectors": image_stats.total_vector_count,
                "user_namespace_exists": user_id in image_stats.namespaces if image_stats.namespaces else False,
                "user_vector_count": image_stats.namespaces.get(user_id, {}).get('vector_count', 0) if image_stats.namespaces and user_id in image_stats.namespaces else 0
            },
            "processing_status": {k: v for k, v in processing_status.items() if k.startswith(user_id)}
        }
    except Exception as e:
        return {"error": str(e)}
