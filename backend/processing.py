import os
import fitz  # This is pymupdf
import io
from PIL import Image
from pathlib import Path
import google.generativeai as genai
import chromadb
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please create a .env file.")

genai.configure(api_key=API_KEY)

# --- Configuration ---
GENERATION_MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

UPLOAD_DIR = Path("uploads")
IMAGE_DIR = Path("images")
CHROMA_DIR = Path("chroma_db")

UPLOAD_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

try:
    generation_model = genai.GenerativeModel(GENERATION_MODEL_NAME)
except Exception as e:
    print(f"Error initializing Google AI generation model: {e}")
    raise

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

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

def process_user_files(user_id: str):
    """
    Processes all uploaded PDFs for a user.
    """
    print(f"Starting processing for user: {user_id}")
    user_upload_dir = UPLOAD_DIR / user_id
    user_image_dir = IMAGE_DIR / user_id
    user_image_dir.mkdir(exist_ok=True)
    if not user_upload_dir.exists():
        print(f"No upload directory found for user {user_id}.")
        return

    text_collection = chroma_client.get_or_create_collection(name=f"{user_id}_text")
    image_collection = chroma_client.get_or_create_collection(name=f"{user_id}_images")
    text_data_for_chroma = {'ids': [], 'embeddings': [], 'documents': [], 'metadatas': []}
    image_data_for_chroma = {'ids': [], 'embeddings': [], 'documents': [], 'metadatas': []}
    text_id_counter = 0
    image_id_counter = 0

    pdf_files = list(user_upload_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found for user {user_id}.")
        return

    try:
        for pdf_path in pdf_files:
            doc = fitz.open(pdf_path)
            print(f"Processing document: {pdf_path.name}")
            for page_num, page in enumerate(doc):
                # 1. Process Text
                page_text = page.get_text()
                if page_text:
                    text_embedding = get_text_embedding(page_text)
                    if text_embedding:
                        text_data_for_chroma['ids'].append(f"text_{text_id_counter}")
                        text_data_for_chroma['embeddings'].append(text_embedding)
                        text_data_for_chroma['documents'].append(page_text)
                        text_data_for_chroma['metadatas'].append({"page_number": page_num + 1, "file_name": pdf_path.name})
                        text_id_counter += 1

                # 2. Process Images
                for img_index, img_info in enumerate(page.get_images(full=True)):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Save image to disk
                    img_filename = f"{pdf_path.stem}_p{page_num+1}_img{img_index}.png"
                    img_path = user_image_dir / img_filename
                    pil_image.save(img_path)

                    # Get image description using Gemini
                    img_desc = get_image_description(pil_image)
                    
                    if img_desc:
                        # Get embedding for the *description*
                        desc_embedding = get_text_embedding(img_desc)
                        if desc_embedding:
                            image_data_for_chroma['ids'].append(f"image_{image_id_counter}")
                            image_data_for_chroma['embeddings'].append(desc_embedding)
                            image_data_for_chroma['documents'].append(img_desc)
                            image_data_for_chroma['metadatas'].append({"img_path": str(img_path), "page_number": page_num + 1, "file_name": pdf_path.name})
                            image_id_counter += 1
            doc.close()

        # Batch add all collected data to ChromaDB
        if text_data_for_chroma['ids']:
            text_collection.add(
                ids=text_data_for_chroma['ids'],
                embeddings=text_data_for_chroma['embeddings'],
                documents=text_data_for_chroma['documents'],
                metadatas=text_data_for_chroma['metadatas']
            )
            print(f"Added {len(text_data_for_chroma['ids'])} text chunks to ChromaDB.")

        if image_data_for_chroma['ids']:
            image_collection.add(
                ids=image_data_for_chroma['ids'],
                embeddings=image_data_for_chroma['embeddings'],
                documents=image_data_for_chroma['documents'],
                metadatas=image_data_for_chroma['metadatas']
            )
            print(f"Added {len(image_data_for_chroma['ids'])} images to ChromaDB.")
        
        print(f"Successfully processed files for user: {user_id}")

    except Exception as e:
        print(f"Error processing files for user {user_id}: {e}")
        pass

# --- NEW Query Logic (Self-Contained) ---

def query_user_data(user_id: str, query: str):
    """
    Queries the user's data using the new Google AI SDK.
    """
    try:
        text_collection = chroma_client.get_collection(name=f"{user_id}_text")
        image_collection = chroma_client.get_collection(name=f"{user_id}_images")
    except Exception as e:
        print(f"Error loading collections for user {user_id}: {e}")
        return "Error: No data found for this user. Please upload and process files first."

    # Step 2: Get embedding for the query
    query_embedding = get_text_embedding(query)
    if not query_embedding:
        return "Error: Could not generate embedding for query."

    # Step 3: Query ChromaDB
    text_results = text_collection.query(query_embeddings=[query_embedding], n_results=5)
    image_results = image_collection.query(query_embeddings=[query_embedding], n_results=5)

    # Step 4: Build context
    final_context_text = "\n".join(text_results['documents'][0])
    
    context_images = []
    image_captions = []
    if image_results and image_results['documents']:
        for metadata, description in zip(image_results['metadatas'][0], image_results['documents'][0]):
            image_path = metadata['img_path']
            try:
                # We need to re-open the image from disk
                img = Image.open(image_path)
                context_images.append(img)
                image_captions.append(f"Caption: {description}\n")
            except FileNotFoundError:
                print(f"Warning: Image file not found at {image_path}. Skipping.")
                pass
    
    # Step 5: Pass context to Gemini
    # Build the prompt in parts for the multimodal model
    prompt_parts = [
        """Instructions: Use the text and images provided as Context to answer the Question.
Think thoroughly before answering. If unsure, respond, "Not enough context to answer".

Context:
 - Text Context:""",
        final_context_text,
        " - Image Context:"
    ]
    
    # Add the images and their captions
    for img, cap in zip(context_images, image_captions):
        prompt_parts.append(img)
        prompt_parts.append(cap)
    
    # Add the final query
    prompt_parts.append(f"\nQuestion:\n{query}\n\nAnswer:")

    try:
        response = generation_model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        print(f"Error generating final response: {e}")
        return f"Error: {e}"