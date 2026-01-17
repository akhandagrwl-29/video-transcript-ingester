import psycopg2
from openai import OpenAI
import os


client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding



conn = psycopg2.connect(NEON_DB_URL)
cur = conn.cursor()

def insert_into_db():
    os.read()

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        cur.execute(
            """
            INSERT INTO youtube_transcripts (video_id, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s)
            """,
            (video_id, i, chunk, embedding)
        )

conn.commit()

if __name__=="__main__":
    insert_into_db()
