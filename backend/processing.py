# sdasdasd
import os
import fitz  # pymupdf
import io
from PIL import Image
from pathlib import Path
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import hashlib
import time
from typing import List, Dict, BinaryIO

# Install with: pip install langchain-text-splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Load Environment Variables ---
load_dotenv()

# API Keys & config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# --- Validation ---
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
GENERATION_MODEL_NAME = "gemini-2.5-flash" # FIXED: Corrected model name
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
EMBEDDING_DIMENSION = 768
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
UPSERT_BATCH = 100
FINAL_QUERY_TOP_K = 12      # How many results to finally use in the context

# Local data directory (e.g., for uploads)
DATA_DIR = Path(os.getenv("DATA_DIR", "."))
# NOTE: TEMP_DIR for images is no longer needed as we upload from memory.

# --- Initialize Google AI ---
try:
    generation_model = genai.GenerativeModel(GENERATION_MODEL_NAME)
except Exception as e:
    print(f"Error initializing Google AI generation model: {e}")
    raise

# --- Initialize Pinecone ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_name_text = "invensight-text"
    index_name_images = "invensight-images"

    existing_indexes = [index.name for index in pc.list_indexes()]

    if index_name_text not in existing_indexes:
        print(f"Creating new Pinecone text index: {index_name_text}")
        pc.create_index(
            name=index_name_text,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
        )

    if index_name_images not in existing_indexes:
        print(f"Creating new Pinecone image index: {index_name_images}")
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

# --- Initialize S3 ---
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

    try:
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
    except ClientError:
        print(f"S3 Bucket {S3_BUCKET_NAME} not found. Creating...")
        s3_client.create_bucket(
            Bucket=S3_BUCKET_NAME,
            CreateBucketConfiguration={'LocationConstraint': AWS_REGION} if AWS_REGION != 'us-east-1' else {}
        )
    print("S3 initialized successfully")
except Exception as e:
    print(f"Error initializing S3: {e}")
    raise

# --- Initialize Text Splitter (Replaces chunk_text) ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)

# --- Utilities ---

def create_preview_text(text: str, limit: int = 800) -> str:
    """Safely creates a preview string from text."""
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + " ..."

def generate_file_hash(file_path: Path) -> str:
    """Generates an MD5 hash for a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def upload_to_s3(file_path: Path, s3_key: str) -> str:
    """Uploads a file from a local path to S3."""
    try:
        s3_client.upload_file(str(file_path), S3_BUCKET_NAME, s3_key)
        s3_uri = f"s3://{S3_BUCKET_NAME}/{s3_key}"
        print(f"Uploaded {file_path.name} to {s3_uri}")
        return s3_uri
    except Exception as e:
        print(f"Error uploading {file_path.name} to S3: {e}")
        return ""

def upload_to_s3_fileobj(file_obj: BinaryIO, s3_key: str) -> str:
    """Uploads a file-like object (in-memory) to S3."""
    try:
        s3_client.upload_fileobj(file_obj, S3_BUCKET_NAME, s3_key)
        s3_uri = f"s3://{S3_BUCKET_NAME}/{s3_key}"
        print(f"Uploaded in-memory object to {s3_uri}")
        return s3_uri
    except Exception as e:
        print(f"Error uploading in-memory object to S3: {e}")
        return ""

def download_from_s3(s3_key: str, local_path: Path) -> bool:
    """Downloads a file from S3 to a local path."""
    try:
        s3_client.download_file(S3_BUCKET_NAME, s3_key, str(local_path))
        print(f"Downloaded {s3_key} from S3")
        return True
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        return False

# --- Embeddings with retries and validation ---
def get_text_embedding(text: str, retries: int = 3, backoff: float = 1.0):
    """Stable embedding generator with retries and dimension check."""
    if not text or not text.strip():
        print("Skipping embedding for empty text.")
        return None
    for attempt in range(1, retries + 1):
        try:
            result = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=text)
            embedding = result.get('embedding')
            if not embedding:
                raise ValueError("No 'embedding' in response from API")
            if len(embedding) != EMBEDDING_DIMENSION:
                raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {len(embedding)}")
            return embedding
        except Exception as e:
            print(f"Embedding attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(backoff * attempt)
    print("Failed to generate embedding after all retries.")
    return None

# --- Image description ---
IMAGE_DESCRIPTION_PROMPT = """Explain what is going on in the image.
If it's a table, extract the table content as text.
If it's a graph, explain the findings in the graph.
Do not invent numbers not present in the image.
Keep description concise (max 200 words).
"""

def get_image_description(pil_image: Image.Image):
    """Generates a text description for a PIL Image."""
    try:
        # FIXED: Comment removed, this is a correct multimodal call.
        response = generation_model.generate_content([IMAGE_DESCRIPTION_PROMPT, pil_image])
        return response.text.strip()
    except Exception as e:
        print(f"Error generating image description: {e}")
        return ""

# --- Processing pipeline ---
def process_user_files(user_id: str, pdf_path: Path):
    """
    Processes a single uploaded PDF:
    - NEW: Checks if file hash already exists to prevent re-processing.
    - Upload PDF to S3
    - Chunk text and embed each chunk with metadata
    - Extract images, describe them, embed descriptions
    - NEW: Upload images from memory to S3
    - Batch upsert vectors to Pinecone
    """
    print(f"Starting processing for user: {user_id}, file: {pdf_path.name}")

    file_hash = generate_file_hash(pdf_path)

    # --- NEW: Re-processing Guard ---
    # Check if this file hash already exists for this user in Pinecone.
    try:
        # Query with a dummy vector but a specific filter.
        check_result = text_index.query(
            vector=[0.0] * EMBEDDING_DIMENSION, 
            top_k=1, 
            filter={"user_id": user_id, "file_hash": file_hash},
            namespace=user_id,
            include_metadata=False # We only care if matches exist
        )
        if check_result.matches:
            print(f"File {pdf_path.name} (hash: {file_hash}) has already been processed for user {user_id}. Skipping.")
            return
    except Exception as e:
        print(f"Warning: Could not check for existing file hash in Pinecone. Proceeding anyway. Error: {e}")
    # --- End Re-processing Guard ---

    s3_key_pdf = f"users/{user_id}/pdfs/{pdf_path.name}"
    s3_uri_pdf = upload_to_s3(pdf_path, s3_key_pdf)
    if not s3_uri_pdf:
        print(f"Failed to upload PDF {pdf_path.name} to S3. Aborting processing for this file.")
        return

    text_vectors = []
    image_vectors = []

    try:
        doc = fitz.open(pdf_path)
        print(f"Opened document: {pdf_path.name}, {len(doc)} pages.")

        for page_num, page in enumerate(doc):
            
            # 1. Text extraction & chunking
            page_text = page.get_text()
            if page_text and page_text.strip():
                # FIXED: Using robust RecursiveCharacterTextSplitter
                chunks = text_splitter.split_text(page_text)
                total_chunks = len(chunks)
                
                for chunk_index, chunk in enumerate(chunks):
                    # FIXED: Embed the *full* anchored chunk
                    anchored_chunk = f"Document: {pdf_path.name}\nPage: {page_num+1}\n{chunk}"
                    emb = get_text_embedding(anchored_chunk)
                    if not emb:
                        continue
                        
                    vector_id = f"{user_id}_{file_hash}_text_p{page_num+1}_c{chunk_index}"
                    metadata = {
                        "user_id": user_id,
                        "file_name": pdf_path.name,
                        "file_hash": file_hash,
                        "s3_uri": s3_uri_pdf,
                        "page_number": page_num + 1,
                        "type": "text",
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                        # FIXED: Use utility for preview, not for embedding
                        "preview": create_preview_text(chunk, 800)
                    }
                    text_vectors.append({"id": vector_id, "values": emb, "metadata": metadata})

            # 2. Image extraction
            for img_index, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image")
                
                if not image_bytes:
                    continue

                try:
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    
                    # Generate description
                    img_desc = get_image_description(pil_image)
                    if not img_desc:
                        print(f"Skipping image p{page_num+1}_i{img_index} due to empty description.")
                        continue

                    # --- FIXED: Upload from memory ---
                    image_buffer = io.BytesIO(image_bytes)
                    img_s3_name = f"{Path(pdf_path.name).stem}_p{page_num+1}_img{img_index}.png"
                    s3_key_img = f"users/{user_id}/images/{img_s3_name}"
                    s3_uri_img = upload_to_s3_fileobj(image_buffer, s3_key_img)
                    # --- End upload from memory ---

                    if s3_uri_img:
                        # FIXED: Embed the *full* anchored description
                        anchored_desc = f"Document: {pdf_path.name}\nPage: {page_num+1}\n{img_desc}"
                        desc_emb = get_text_embedding(anchored_desc)
                        
                        if desc_emb:
                            vector_id = f"{user_id}_{file_hash}_img_p{page_num+1}_i{img_index}"
                            metadata = {
                                "user_id": user_id,
                                "file_name": pdf_path.name,
                                "file_hash": file_hash,
                                "s3_uri_pdf": s3_uri_pdf,
                                "s3_uri_image": s3_uri_img,
                                "page_number": page_num + 1,
                                "type": "image",
                                "image_index": img_index,
                                # FIXED: Use utility for preview
                                "description_preview": create_preview_text(img_desc, 800)
                            }
                            image_vectors.append({"id": vector_id, "values": desc_emb, "metadata": metadata})
                
                except Exception as img_e:
                    print(f"Error processing image p{page_num+1}_i{img_index}: {img_e}")
                    # Continue to next image

        # close doc
        doc.close()

        # Batch upsert helper
        def batch_upsert(index, items, item_type="vectors"):
            for i in range(0, len(items), UPSERT_BATCH):
                batch = items[i:i + UPSERT_BATCH]
                try:
                    index.upsert(vectors=batch, namespace=user_id)
                    print(f"Upserted batch {i//UPSERT_BATCH + 1} of {item_type} to Pinecone")
                except Exception as e:
                    print(f"Error upserting batch: {e}")

        if text_vectors:
            batch_upsert(text_index, text_vectors, "text")
            print(f"Uploaded {len(text_vectors)} text vectors to Pinecone")

        if image_vectors:
            batch_upsert(image_index, image_vectors, "image")
            print(f"Uploaded {len(image_vectors)} image vectors to Pinecone")

        print(f"Finished processing file for user: {user_id}")

    except Exception as e:
        print(f"Error processing file {pdf_path.name} for user {user_id}: {e}")
        # Optionally, you could try to clean up the S3 PDF upload here if processing fails
        raise

# --- Querying pipeline ---
def query_user_data(user_id: str, query: str, top_k: int = FINAL_QUERY_TOP_K):
    """
    Queries Pinecone (text + image indices), merges and reranks by Pinecone scores,
    constructs a structured prompt (text + image captions + s3 links), and asks Gemini.
    """
    query_emb = get_text_embedding(query)
    if not query_emb:
        return "Error: Could not generate embedding for query."

    try:
        # FIXED: Query both indexes for the *full* amount (top_k)
        text_results = text_index.query(
            vector=query_emb, 
            top_k=top_k, 
            namespace=user_id, 
            include_metadata=True
        )
        image_results = image_index.query(
            vector=query_emb, 
            top_k=top_k, 
            namespace=user_id, 
            include_metadata=True
        )

        print(f"Text matches retrieved: {len(text_results.matches)}; Image matches retrieved: {len(image_results.matches)}")

        if not text_results.matches and not image_results.matches:
            return ("I couldn't find any relevant information in your documents.\n\n"
                    "This might be because:\n"
                    "1. Your document is still being processed.\n"
                    "2. You haven't uploaded any documents yet.\n"
                    "3. The query doesn't match your document content.")

        # Merge results and sort by score
        merged = []
        for m in (text_results.matches or []):
            merged.append((m.score, "text", m.metadata))
        for m in (image_results.matches or []):
            merged.append((m.score, "image", m.metadata))
        
        # Sort by score (descending) to get the most relevant items
        merged.sort(key=lambda x: x[0], reverse=True)

        # Build final context using top N merged items
        context_parts = []
        # FIXED: Slice the *merged and sorted* list to get the absolute best results
        for score, typ, meta in merged[:top_k]:
            if typ == "text":
                preview = meta.get("preview") or ""
                context_parts.append(
                    f"[Text Chunk] (Relevance: {score:.3f})\n"
                    f"Source: {meta.get('file_name')}, Page: {meta.get('page_number')}\n"
                    f"{preview}"
                )
            else:
                desc = meta.get("description_preview") or ""
                s3_img = meta.get("s3_uri_image")
                context_parts.append(
                    f"[Image] (Relevance: {score:.3f})\n"
                    f"Source: {meta.get('file_name')}, Page: {meta.get('page_number')}\n"
                    f"Image S3 Link: {s3_img}\n"
                    f"Description: {desc}"
                )

        final_context = "\n\n---\n\n".join(context_parts)

        # FIXED: Stricter prompt for better RAG performance
        prompt = f"""
You are an assistant. Your task is to answer the user's QUESTION based *only* on the provided CONTEXT.
Do not use any outside knowledge.
If the answer is not contained within the CONTEXT, state: "I could not find an answer to that in the provided documents."

CONTEXT:
{final_context}

QUESTION:
{query}

FINAL ANSWER:
"""
        # Ask Gemini
        response = generation_model.generate_content([prompt])
        return response.text

    except Exception as e:
        print(f"Error querying data for user {user_id}: {e}")
        return f"An error occurred while querying: {e}"

# --- File listing & deletion (Robustified) ---
def list_user_files(user_id: str) -> List[Dict]:
    """Lists all PDF files for a user from S3."""
    try:
        prefix = f"users/{user_id}/pdfs/"
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                # Ignore "directory" objects if S3 creates them
                if obj['Size'] > 0 and obj['Key'].endswith('.pdf'):
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
    """
    Deletes a file and all its associated data from S3 (PDF, images)
    and Pinecone (text vectors, image vectors) using metadata filters.
    """
    print(f"Attempting to delete all data for: {file_name} for user: {user_id}")
    try:
        # 1. Delete PDF from S3
        s3_key_pdf = f"users/{user_id}/pdfs/{file_name}"
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key_pdf)
        print(f"Deleted PDF from S3: {s3_key_pdf}")

        # 2. Delete all associated images from S3
        # This lists all images that *started with* the PDF's stem
        image_prefix = f"users/{user_id}/images/{Path(file_name).stem}"
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=image_prefix)
        if 'Contents' in response:
            objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
            if objects_to_delete:
                s3_client.delete_objects(Bucket=S3_BUCKET_NAME, Delete={'Objects': objects_to_delete})
                print(f"Deleted {len(objects_to_delete)} associated images from S3.")

        # 3. Delete from Pinecone using metadata filter (the clean way)
        # This avoids needing to know file_hash, just user_id and file_name.
        delete_filter = {"user_id": user_id, "file_name": file_name}
        
        text_index.delete(filter=delete_filter, namespace=user_id)
        print(f"Sent delete request to text index for: {file_name}")
        
        image_index.delete(filter=delete_filter, namespace=user_id)
        print(f"Sent delete request to image index for: {file_name}")

        print(f"Successfully deleted all artifacts for file {file_name} for user {user_id}")
        return True
    except Exception as e:
        print(f"Error deleting file {file_name}: {e}")
        return False

# --- Example Usage (to be run in your application) ---
if __name__ == "__main__":
    # This is a conceptual test.
    # In a real app, you'd get user_id from a session and file_path from an upload.
    
    print("Running RAG pipeline example...")
    TEST_USER_ID = "example_user_123"
    
    # --- Create a dummy PDF for testing ---
    # (Requires `pip install reportlab`)
    try:
        import importlib
        # Dynamically import reportlab modules to avoid static analysis import errors
        reportlab_canvas = importlib.import_module('reportlab.pdfgen.canvas')
        canvas = reportlab_canvas
        pagesizes = importlib.import_module('reportlab.lib.pagesizes')
        LETTER = pagesizes.LETTER
        
        DUMMY_PDF_PATH = DATA_DIR / "dummy_test_doc.pdf"
        
        c = canvas.Canvas(str(DUMMY_PDF_PATH), pagesize=LETTER)
        c.drawString(72, 800, "InvenSight Test Document - Page 1")
        c.drawString(72, 780, "This is the first paragraph. It talks about cloud infrastructure.")
        c.drawString(72, 760, "RAG stands for Retrieval-Augmented Generation. It combines search with LLMs.")
        c.showPage()
        c.drawString(72, 800, "InvenSight Test Document - Page 2")
        c.drawString(72, 780, "This is the second page. It discusses vector databases.")
        c.drawString(72, 760, "Pinecone, Weaviate, and Milvus are popular vector DBs.")
        c.save()
        print(f"Created dummy PDF: {DUMMY_PDF_PATH}")

        # 1. Process the file
        # Running it twice to show the re-processing guard works
        print("\n--- Processing Pass 1 ---")
        process_user_files(TEST_USER_ID, DUMMY_PDF_PATH)
        
        print("\n--- Processing Pass 2 (Should skip) ---")
        process_user_files(TEST_USER_ID, DUMMY_PDF_PATH)
        
        # Give Pinecone a moment to index
        print("\nWaiting 10s for indexing...")
        time.sleep(10) 

        # 2. List files
        print("\n--- Listing Files ---")
        files = list_user_files(TEST_USER_ID)
        print(f"Found files: {files}")

        # 3. Query
        print("\n--- Querying Data ---")
        query1 = "What is RAG?"
        answer1 = query_user_data(TEST_USER_ID, query1)
        print(f"Q: {query1}\nA: {answer1}")
        
        query2 = "What are popular vector DBs?"
        answer2 = query_user_data(TEST_USER_ID, query2)
        print(f"Q: {query2}\nA: {answer2}")

        query3 = "What is the weather today?" # Should fail
        answer3 = query_user_data(TEST_USER_ID, query3)
        print(f"Q: {query3}\nA: {answer3}")

        # 4. Delete
        print("\n--- Deleting File ---")
        delete_user_file(TEST_USER_ID, DUMMY_PDF_PATH.name)
        
        print("\nWaiting 5s for deletion to propagate...")
        time.sleep(5)
        
        files_after_delete = list_user_files(TEST_USER_ID)
        print(f"Files after delete: {files_after_delete}")
        
        # 5. Query after delete
        print("\n--- Querying After Delete ---")
        answer4 = query_user_data(TEST_USER_ID, query1)
        print(f"Q: {query1}\nA: {answer4}")
        
        # Cleanup dummy file
        DUMMY_PDF_PATH.unlink(missing_ok=True)

    except ImportError:
        print("\nSkipping `if __name__ == '__main__'` block: `reportlab` not installed.")
        print("Install it with `pip install reportlab` to run the example.")
    except Exception as e:
        print(f"\nAn error occurred during the example run: {e}")