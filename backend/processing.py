import os
import fitz  # This is pymupdf
import io
from PIL import Image
from pathlib import Path
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import hashlib
import json
from typing import List, Dict
import tempfile  # Add this with other imports

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file.")
if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    raise ValueError("AWS credentials not found in .env file.")
if not S3_BUCKET_NAME:
    raise ValueError("S3_BUCKET_NAME not found in .env file.")

# Configure Google AI
genai.configure(api_key=GEMINI_API_KEY)

# --- Configuration ---
GENERATION_MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
EMBEDDING_DIMENSION = 768  # Google's text-embedding-004 dimension

# Local temp directory for processing (will be cleaned up)
DATA_DIR = Path(os.getenv("DATA_DIR", "."))
TEMP_DIR = DATA_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Google AI
try:
    generation_model = genai.GenerativeModel(GENERATION_MODEL_NAME)
except Exception as e:
    print(f"Error initializing Google AI generation model: {e}")
    raise

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create indexes if they don't exist
    index_name_text = "invensight-text"
    index_name_images = "invensight-images"
    
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name_text not in existing_indexes:
        pc.create_index(
            name=index_name_text,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
        )
    
    if index_name_images not in existing_indexes:
        pc.create_index(
            name=index_name_images,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
        )
    
    text_index = pc.Index(index_name_text)
    image_index = pc.Index(index_name_images)
    
    print("Pinecone initialized successfully")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    raise

# Initialize S3
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
    # Check if bucket exists, create if not
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
    except ClientError:
        s3_client.create_bucket(
            Bucket=S3_BUCKET_NAME,
            CreateBucketConfiguration={'LocationConstraint': AWS_REGION} if AWS_REGION != 'us-east-1' else {}
        )
    
    print("S3 initialized successfully")
except Exception as e:
    print(f"Error initializing S3: {e}")
    raise

# --- Image Description Prompt ---
IMAGE_DESCRIPTION_PROMPT = """Explain what is going on in the image.
If it's a table, extract all elements of the table.
If it's a graph, explain the findings in the graph.
Do not include any numbers that are not mentioned in the image.
"""

def get_image_description(pil_image):
    """Generates a text description for a PIL image."""
    try:
        response = generation_model.generate_content([IMAGE_DESCRIPTION_PROMPT, pil_image])
        return response.text
    except Exception as e:
        print(f"Error generating image description: {e}")
        return ""

def get_text_embedding(text):
    """Generates an embedding for a piece of text."""
    try:
        result = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=text)
        return result['embedding']
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        return None

def upload_to_s3(file_path: Path, s3_key: str) -> str:
    """Uploads a file to S3 and returns the S3 URI."""
    try:
        s3_client.upload_file(str(file_path), S3_BUCKET_NAME, s3_key)
        s3_uri = f"s3://{S3_BUCKET_NAME}/{s3_key}"
        print(f"Uploaded {file_path.name} to {s3_uri}")
        return s3_uri
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return ""

def download_from_s3(s3_key: str, local_path: Path) -> bool:
    """Downloads a file from S3 to local path."""
    try:
        s3_client.download_file(S3_BUCKET_NAME, s3_key, str(local_path))
        print(f"Downloaded {s3_key} from S3")
        return True
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        return False

def generate_file_hash(file_path: Path) -> str:
    """Generates MD5 hash of a file to track if it's already processed."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def delete_user_embeddings(user_id: str):
    """Deletes all embeddings for a user from Pinecone."""
    try:
        # Delete from text index
        text_index.delete(filter={"user_id": user_id})
        # Delete from image index
        image_index.delete(filter={"user_id": user_id})
        print(f"Deleted all embeddings for user: {user_id}")
    except Exception as e:
        print(f"Error deleting embeddings: {e}")

def process_user_files(user_id: str, pdf_path: Path):
    """
    Processes a single uploaded PDF for a user.
    Stores PDF in S3 and embeddings in Pinecone.
    """
    print(f"Starting processing for user: {user_id}, file: {pdf_path.name}")
    
    # Generate file hash to avoid duplicate processing
    file_hash = generate_file_hash(pdf_path)
    
    # Upload PDF to S3
    s3_key_pdf = f"users/{user_id}/pdfs/{pdf_path.name}"
    s3_uri_pdf = upload_to_s3(pdf_path, s3_key_pdf)
    if not s3_uri_pdf:
        print(f"Failed to upload PDF to S3")
        return

    # Prepare data for batch upsert
    text_vectors = []
    image_vectors = []

    try:
        doc = fitz.open(pdf_path)
        print(f"Processing document: {pdf_path.name}")
        
        for page_num, page in enumerate(doc):
            # 1. Process Text
            page_text = page.get_text()
            if page_text.strip():
                text_embedding = get_text_embedding(page_text)
                if text_embedding:
                    vector_id = f"{user_id}_{file_hash}_text_p{page_num+1}"
                    metadata = {
                        "user_id": user_id,
                        "file_name": pdf_path.name,
                        "file_hash": file_hash,
                        "s3_uri": s3_uri_pdf,
                        "page_number": page_num + 1,
                        "type": "text",
                        "content": page_text[:1000]  # Store preview (Pinecone metadata limit)
                    }
                    text_vectors.append({
                        "id": vector_id,
                        "values": text_embedding,
                        "metadata": metadata
                    })

            # 2. Process Images
            for img_index, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes))
                
                # Save image temporarily
                temp_img_path = TEMP_DIR / f"{pdf_path.stem}_p{page_num+1}_img{img_index}.png"
                pil_image.save(temp_img_path)
                
                # Upload image to S3
                s3_key_img = f"users/{user_id}/images/{temp_img_path.name}"
                s3_uri_img = upload_to_s3(temp_img_path, s3_key_img)
                
                # Get image description using Gemini
                img_desc = get_image_description(pil_image)
                
                if img_desc and s3_uri_img:
                    # Get embedding for the description
                    desc_embedding = get_text_embedding(img_desc)
                    if desc_embedding:
                        vector_id = f"{user_id}_{file_hash}_img_p{page_num+1}_i{img_index}"
                        metadata = {
                            "user_id": user_id,
                            "file_name": pdf_path.name,
                            "file_hash": file_hash,
                            "s3_uri_pdf": s3_uri_pdf,
                            "s3_uri_image": s3_uri_img,
                            "page_number": page_num + 1,
                            "type": "image",
                            "description": img_desc[:1000]  # Store preview
                        }
                        image_vectors.append({
                            "id": vector_id,
                            "values": desc_embedding,
                            "metadata": metadata
                        })
                
                # Clean up temp image
                temp_img_path.unlink(missing_ok=True)
        
        doc.close()

        # Batch upsert to Pinecone
        if text_vectors:
            text_index.upsert(vectors=text_vectors, namespace=user_id)
            print(f"Uploaded {len(text_vectors)} text embeddings to Pinecone")
        
        if image_vectors:
            image_index.upsert(vectors=image_vectors, namespace=user_id)
            print(f"Uploaded {len(image_vectors)} image embeddings to Pinecone")
        
        print(f"Successfully processed file for user: {user_id}")

    except Exception as e:
        print(f"Error processing file for user {user_id}: {e}")
        raise

def query_user_data(user_id: str, query: str):
    """
    Queries the user's data using Pinecone and S3.
    """
    query_embedding = get_text_embedding(query)
    if not query_embedding:
        return "Error: Could not generate embedding for query."

    try:
        # Query Pinecone
        text_results = text_index.query(
            vector=query_embedding,
            top_k=10,  # ← Increased from 5
            namespace=user_id,
            include_metadata=True
        )
        
        image_results = image_index.query(
            vector=query_embedding,
            top_k=10,  # ← Increased from 5
            namespace=user_id,
            include_metadata=True
        )
        
        # Debug: Check if we got any results
        print(f"Text results: {len(text_results.matches)} matches")
        print(f"Image results: {len(image_results.matches)} matches")
        
        # Check if user has ANY data
        if not text_results.matches and not image_results.matches:
            return """I don't have any documents to search through yet. 

Possible reasons:
1. Your document is still being processed (this can take 30-60 seconds)
2. No documents have been uploaded yet
3. The document upload may have failed

Please wait a moment and try again, or upload a new document."""
        
        # Build context from text results
        text_contexts = []
        for match in text_results.matches:
            print(f"Text match score: {match.score}")  # Debug
            if match.metadata.get('content'):
                text_contexts.append(match.metadata['content'])
        
        final_context_text = "\n\n".join(text_contexts) if text_contexts else "No relevant text found."
        
        # Build context from image results
        context_images = []
        image_captions = []
        
        for match in image_results.matches:
            print(f"Image match score: {match.score}")  # Debug
            s3_uri_image = match.metadata.get('s3_uri_image')
            description = match.metadata.get('description', '')
            
            if s3_uri_image:
                s3_key = s3_uri_image.replace(f"s3://{S3_BUCKET_NAME}/", "")
                temp_img_path = TEMP_DIR / f"query_{s3_key.split('/')[-1]}"
                if download_from_s3(s3_key, temp_img_path):
                    try:
                        with Image.open(temp_img_path) as img:
                            img_copy = img.copy()
                        context_images.append(img_copy)
                        image_captions.append(f"Caption: {description}\n")
                    except Exception as e:
                        print(f"Error opening image: {e}")
                    finally:
                        temp_img_path.unlink(missing_ok=True)
        
        # Build prompt
        prompt_parts = [
            """Instructions: Use the text and images provided as Context to answer the Question.
Think thoroughly before answering. Try to provide a helpful answer even if the context is limited.
Only respond "Not enough context to answer" if there is absolutely NO relevant information.

Context:
 - Text Context:""",
            final_context_text,
            " - Image Context:"
        ]
        
        for img, cap in zip(context_images, image_captions):
            prompt_parts.append(img)
            prompt_parts.append(cap)
        
        prompt_parts.append(f"\nQuestion:\n{query}\n\nAnswer:")

        # Generate response
        response = generation_model.generate_content(prompt_parts)
        return response.text
            
    except Exception as e:
        print(f"Error querying data for user {user_id}: {e}")
        return f"Error: {e}"

def list_user_files(user_id: str) -> List[Dict]:
    """Lists all files uploaded by a user from S3."""
    try:
        prefix = f"users/{user_id}/pdfs/"
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
        
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                files.append({
                    "file_name": obj['Key'].split('/')[-1],
                    "s3_key": obj['Key'],
                    "size": obj['Size'],
                    "last_modified": obj['LastModified'].isoformat()
                })
        return files
    except Exception as e:
        print(f"Error listing files: {e}")
        return []

def delete_user_file(user_id: str, file_name: str):
    """Deletes a specific file and its embeddings."""
    try:
        # Delete PDF from S3
        s3_key = f"users/{user_id}/pdfs/{file_name}"
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        
        # Delete associated images from S3
        prefix = f"users/{user_id}/images/{Path(file_name).stem}"
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
        if 'Contents' in response:
            for obj in response['Contents']:
                s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=obj['Key'])
        
        # Delete embeddings from Pinecone (filter by file_name)
        text_index.delete(filter={"user_id": user_id, "file_name": file_name}, namespace=user_id)
        image_index.delete(filter={"user_id": user_id, "file_name": file_name}, namespace=user_id)
        
        print(f"Deleted file {file_name} for user {user_id}")
        return True
    except Exception as e:
        print(f"Error deleting file: {e}")
        return False