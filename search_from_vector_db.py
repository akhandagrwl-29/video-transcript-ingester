import psycopg2
import os
import sys
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Make sure to set environment variables manually.")

# Try to import pgvector for proper vector type handling
try:
    import pgvector.psycopg2
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    print("Warning: pgvector package not found. Using string format for vectors.")

# Initialize SentenceTransformer model (same as in insert script)
try:
    _embedding_model = SentenceTransformer('all-mpnet-base-v2')
    print("Loaded embedding model: all-mpnet-base-v2 (768 dimensions)")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    _embedding_model = None


def get_embedding(text: str) -> List[float]:
    """Generate embedding for the given text using SentenceTransformer."""
    if _embedding_model is None:
        raise RuntimeError("Embedding model not initialized")
    
    embedding = _embedding_model.encode(text, convert_to_numpy=True)
    embedding_list = embedding.tolist()
    
    # Pad to 1536 dimensions to match schema
    if len(embedding_list) != 1536:
        if len(embedding_list) < 1536:
            embedding_list = embedding_list + [0.0] * (1536 - len(embedding_list))
        else:
            embedding_list = embedding_list[:1536]
    
    return embedding_list


def get_db_connection():
    """Get database connection using environment variables."""
    PSQL_USERNAME = os.environ.get("PSQL_USERNAME")
    PSQL_PASSWORD = os.environ.get("PSQL_PASSWORD")
    PSQL_HOST = os.environ.get("PSQL_HOST")
    
    if not PSQL_USERNAME:
        raise ValueError("PSQL_USERNAME environment variable is not set")
    if not PSQL_PASSWORD:
        raise ValueError("PSQL_PASSWORD environment variable is not set")
    if not PSQL_HOST:
        raise ValueError("PSQL_HOST environment variable is not set")
    
    neon_db_url = f'postgresql://{PSQL_USERNAME}:{PSQL_PASSWORD}@{PSQL_HOST}/neondb?sslmode=require&channel_binding=require'
    
    conn = psycopg2.connect(neon_db_url)
    
    # Register pgvector adapter if available
    if PGVECTOR_AVAILABLE:
        pgvector.psycopg2.register_vector(conn)
    
    return conn


def search_vector_db(query_text: str, top_k: int = 5) -> List[Tuple[str, str, int, float, str]]:
    """
    Search the vector database for similar transcript chunks.
    
    Args:
        query_text: The search query text
        top_k: Number of top results to return (default: 5)
    
    Returns:
        List of tuples: (video_id, content, chunk_index, similarity_score, full_content_preview)
    """
    # Generate embedding for the query
    query_embedding = get_embedding(query_text)
    
    # Connect to database
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Format embedding for database query
        if PGVECTOR_AVAILABLE:
            # Use pgvector adapter
            embedding_value = query_embedding
        else:
            # Format as string array
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            embedding_value = embedding_str
        
        # Perform cosine similarity search using pgvector
        # Using 1 - cosine_distance for similarity (higher is better)
        cur.execute(
            """
            SELECT 
                video_id,
                content,
                chunk_index,
                1 - (embedding <=> %s::vector) as similarity,
                content
            FROM youtube_transcripts
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding_value, embedding_value, top_k)
        )
        
        results = cur.fetchall()
        
        return results
    
    finally:
        cur.close()
        conn.close()


def format_results(results: List[Tuple[str, str, int, float, str]], query: str):
    """Format and display search results."""
    if not results:
        print(f"\nNo results found for query: '{query}'")
        return
    
    print(f"\n{'='*80}")
    print(f"Search Results for: '{query}'")
    print(f"{'='*80}\n")
    
    for i, (video_id, content, chunk_index, similarity, _) in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Video ID: {video_id}")
        print(f"  Chunk Index: {chunk_index}")
        print(f"  Similarity Score: {similarity:.4f}")
        print(f"  Content Preview: {content[:200]}..." if len(content) > 200 else f"  Content: {content}")
        print(f"  {'-'*78}\n")


def main():
    """Main function to run the search."""
    if len(sys.argv) < 2:
        print("Usage: python search_from_vector_db.py '<search query>' [top_k]")
        print("Example: python search_from_vector_db.py 'database optimization' 10")
        sys.exit(1)
    
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print(f"Searching for: '{query}'")
    print(f"Returning top {top_k} results...")
    
    try:
        results = search_vector_db(query, top_k=top_k)
        format_results(results, query)
        
        # Also return results for programmatic use
        return results
    
    except Exception as e:
        print(f"Error during search: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
