import psycopg2
from openai import OpenAI
import os
import glob
from typing import List
from sentence_transformers import SentenceTransformer
# Load environment variables from .env file
# #region agent log
import json
log_path = "/debug.log"
def debug_log(location, message, data, hypothesis_id=None):
    try:
        with open(log_path, "a") as f:
            log_entry = {
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(__import__("time").time() * 1000)
            }
            f.write(json.dumps(log_entry) + "\n")
    except: pass
# #endregion

try:
    from dotenv import load_dotenv
    # #region agent log
    debug_log("insert_into_vector_db.py:15", "Loading .env file", {"dotenv_available": True}, "B")
    result = load_dotenv()
    debug_log("insert_into_vector_db.py:16", "load_dotenv() result", {"result": result, "env_file_exists": __import__("os").path.exists(".env")}, "B")
    # #endregion
    load_dotenv()
except ImportError:
    # #region agent log
    debug_log("insert_into_vector_db.py:19", "python-dotenv not available", {}, "B")
    # #endregion
    print("Warning: python-dotenv not installed. Make sure to set environment variables manually.")

# Try to import pgvector for proper vector type handling
try:
    import pgvector.psycopg2
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    print("Warning: pgvector package not found. Using string format for vectors.")


# client = OpenAI()

# Initialize SentenceTransformer model once (not on every call)
# Using 'all-mpnet-base-v2' produces 768 dimensions
# For 1536 dimensions, we'll use a model that supports it or pad/concatenate
try:
    # Try to use a model that produces 1536 dimensions, or use a larger model
    # 'sentence-transformers/all-mpnet-base-v2' = 768 dims
    # 'sentence-transformers/all-MiniLM-L6-v2' = 384 dims
    # For 1536, we can use OpenAI's model or concatenate two 768-dim embeddings
    # For now, using a model that we'll need to handle dimension mismatch
    _embedding_model = SentenceTransformer('all-mpnet-base-v2')
    print("Loaded embedding model: all-mpnet-base-v2 (768 dimensions)")
    print("Note: Schema expects 1536 dimensions. Consider using OpenAI embeddings or adjusting schema.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    _embedding_model = None

def get_embedding(text: str) -> List[float]:
    """Generate embedding for the given text using SentenceTransformer."""
    if _embedding_model is None:
        raise RuntimeError("Embedding model not initialized")
    
    embedding = _embedding_model.encode(text, convert_to_numpy=True)
    
    # Convert to list and handle dimension mismatch
    embedding_list = embedding.tolist()
    
    # #region agent log
    debug_log("insert_into_vector_db.py:75", "Generated embedding", {
        "embedding_dim": len(embedding_list),
        "expected_dim": 1536,
        "dimension_match": len(embedding_list) == 1536
    }, "C")
    # #endregion
    
    # If dimension mismatch, pad or truncate to match schema (1536)
    if len(embedding_list) != 1536:
        if len(embedding_list) < 1536:
            # Pad with zeros to reach 1536 dimensions
            embedding_list = embedding_list + [0.0] * (1536 - len(embedding_list))
        else:
            # Truncate to 1536 dimensions
            embedding_list = embedding_list[:1536]
    
    return embedding_list


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at a space or sentence boundary near the end
        if end < len(text):
            # Look for sentence endings first
            for punct in ['. ', '! ', '? ', '\n']:
                last_punct = chunk.rfind(punct)
                if last_punct > chunk_size * 0.7:  # Only break if we're past 70% of chunk
                    chunk = chunk[:last_punct + 1]
                    end = start + len(chunk)
                    break
            else:
                # If no sentence boundary, break at space
                last_space = chunk.rfind(' ')
                if last_space > chunk_size * 0.7:
                    chunk = chunk[:last_space]
                    end = start + len(chunk)
        
        chunks.append(chunk.strip())
        start = end - overlap  # Overlap for context
    
    return chunks


def extract_video_id(filename: str) -> str:
    """Extract video ID from transcript filename."""
    # Remove directory path and .txt extension
    basename = os.path.basename(filename)
    video_id = os.path.splitext(basename)[0]
    return video_id


def format_embedding_for_db(embedding: List[float]) -> str:
    """
    Format embedding list as a string for PostgreSQL VECTOR type.
    pgvector expects the format: [1,2,3,...]
    """
    return "[" + ",".join(str(x) for x in embedding) + "]"



def insert_into_db():
    """Read all transcript files and insert them into the vector database."""
    # #region agent log
    debug_log("insert_into_vector_db.py:96", "insert_into_db() entry", {}, "A")
    # #endregion
    
    # Get database credentials from environment variables
    PSQL_USERNAME = os.environ.get("PSQL_USERNAME")
    PSQL_PASSWORD = os.environ.get("PSQL_PASSWORD")
    PSQL_HOST = os.environ.get("PSQL_HOST")
    
    # #region agent log
    debug_log("insert_into_vector_db.py:101", "Environment variables read", {
        "PSQL_USERNAME": PSQL_USERNAME if PSQL_USERNAME else None,
        "PSQL_PASSWORD": "***" if PSQL_PASSWORD else None,
        "PSQL_HOST": PSQL_HOST if PSQL_HOST else None,
        "username_is_placeholder": PSQL_USERNAME == "your-username-here" if PSQL_USERNAME else None,
        "host_is_placeholder": PSQL_HOST == "your-host-here" if PSQL_HOST else None
    }, "A")
    # #endregion
    
    # Validate that all required environment variables are set
    if not PSQL_USERNAME:
        # #region agent log
        debug_log("insert_into_vector_db.py:104", "PSQL_USERNAME validation failed", {"value": PSQL_USERNAME}, "A")
        # #endregion
        raise ValueError("PSQL_USERNAME environment variable is not set")
    if not PSQL_PASSWORD:
        # #region agent log
        debug_log("insert_into_vector_db.py:107", "PSQL_PASSWORD validation failed", {"value": "***" if PSQL_PASSWORD else None}, "A")
        # #endregion
        raise ValueError("PSQL_PASSWORD environment variable is not set")
    if not PSQL_HOST:
        # #region agent log
        debug_log("insert_into_vector_db.py:109", "PSQL_HOST validation failed", {"value": PSQL_HOST}, "A")
        # #endregion
        raise ValueError("PSQL_HOST environment variable is not set")
    
    # #region agent log
    debug_log("insert_into_vector_db.py:112", "All env vars validated", {
        "username_length": len(PSQL_USERNAME) if PSQL_USERNAME else 0,
        "password_length": len(PSQL_PASSWORD) if PSQL_PASSWORD else 0,
        "host_length": len(PSQL_HOST) if PSQL_HOST else 0
    }, "A")
    # #endregion
    
    # Construct database URL
    neon_db_url = f'postgresql://{PSQL_USERNAME}:{PSQL_PASSWORD}@{PSQL_HOST}/neondb?sslmode=require&channel_binding=require'
    
    # #region agent log
    debug_log("insert_into_vector_db.py:115", "Connection URL constructed", {
        "url_masked": f'postgresql://{PSQL_USERNAME}:***@{PSQL_HOST}/neondb?sslmode=require&channel_binding=require',
        "host_in_url": PSQL_HOST
    }, "A")
    # #endregion
    
    # Connect to database
    # #region agent log
    debug_log("insert_into_vector_db.py:118", "Attempting database connection", {"host": PSQL_HOST}, "A")
    # #endregion
    try:
        conn = psycopg2.connect(neon_db_url)
        # #region agent log
        debug_log("insert_into_vector_db.py:120", "Database connection successful", {}, "A")
        # #endregion
    except Exception as e:
        # #region agent log
        debug_log("insert_into_vector_db.py:123", "Database connection failed", {"error_type": type(e).__name__, "error_message": str(e)}, "A")
        # #endregion
        raise
    
    # Register pgvector adapter if available
    if PGVECTOR_AVAILABLE:
        pgvector.psycopg2.register_vector(conn)
    
    cur = conn.cursor()
    
    # Get all transcript files
    transcript_dir = "transcripts"
    transcript_files = glob.glob(os.path.join(transcript_dir, "*.txt"))
    
    if not transcript_files:
        print(f"No transcript files found in {transcript_dir}/")
        return
    
    print(f"Found {len(transcript_files)} transcript files to process")
    
    total_chunks = 0
    
    for transcript_file in transcript_files:
        video_id = extract_video_id(transcript_file)
        print(f"\nProcessing video: {video_id}")
        
        try:
            # Read transcript file
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript_text = f.read().strip()
            
            if not transcript_text:
                print(f"  Skipping {video_id}: empty transcript")
                continue
            
            # Chunk the transcript
            chunks = chunk_text(transcript_text)
            print(f"  Split into {len(chunks)} chunks")
            
            # Insert each chunk into database
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                try:
                    # Generate embedding
                    embedding = get_embedding(chunk)
                    
                    # Verify embedding dimension matches schema (1536)
                    if len(embedding) != 1536:
                        print(f"    Warning: Embedding dimension is {len(embedding)}, expected 1536")
                    
                    # Format embedding for database insertion
                    if PGVECTOR_AVAILABLE:
                        # Use pgvector adapter if available
                        embedding_value = embedding
                    else:
                        # Format as string array for pgvector
                        embedding_value = format_embedding_for_db(embedding)
                    
                    # Insert into database
                    cur.execute(
                        """
                        INSERT INTO youtube_transcripts (video_id, chunk_index, content, embedding)
                        VALUES (%s, %s, %s, %s::vector)
                        """,
                        (video_id, i, chunk, embedding_value)
                    )
                    
                    total_chunks += 1
                    if (i + 1) % 10 == 0:
                        print(f"    Inserted {i + 1}/{len(chunks)} chunks...")
                
                except Exception as e:
                    print(f"    Error processing chunk {i} for {video_id}: {e}")
                    continue
            
            print(f"  ✓ Completed {video_id}: {len(chunks)} chunks inserted")
        
        except Exception as e:
            print(f"  ✗ Error processing {video_id}: {e}")
            continue
    
    # Commit all changes
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"\n✓ All done! Inserted {total_chunks} total chunks into the database.")


if __name__ == "__main__":
    insert_into_db()
