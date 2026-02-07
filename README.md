# Video Transcript Ingester

A simple project that helps you **find the exact YouTube video** inside a playlist that best answers your search query.

Instead of manually watching multiple videos, this tool ingests video transcripts from a YouTube playlist and matches them against your search input to recommend the most relevant video.

---

## 🚀 What It Does

* Takes a **search query** (your question or keyword)
* Takes a **YouTube playlist link**
* Fetches and processes transcripts of videos in the playlist
* Matches your query against the transcripts
* Recommends the **exact video** where the answer is most relevant

---

## 🧠 Use Case

Perfect for:

* Educational playlists (DSA, system design, tutorials)
* Long playlists where answers are scattered across videos
* Quickly finding *which video* explains a specific concept

---

## 🛠️ How It Works (High Level)

1. Extract video IDs from the playlist
2. Fetch transcripts for each video
3. Index / analyze transcript text
4. Compare search query with transcripts
5. Rank videos based on relevance
6. Return the best matching video(s)

---

## 📦 Input

* **Search Query** – what you’re looking for
* **YouTube Playlist URL** – playlist to search within

---

## 📤 Output

* Recommended YouTube video link
* (Optionally) relevance score or matched transcript snippet

---

## 🧪 Example

```text
Search Query: "binary search tree deletion"
Playlist: https://www.youtube.com/playlist?list=XXXX

Result:
→ Video: "BST Deletion Explained Clearly"
```

---

## 🔮 Future Improvements

* Timestamp-level recommendations
* Multiple top matches instead of one
* UI / Web interface
* Support for non-English transcripts
* Better semantic search (embeddings)

---

## 🤝 Contributing

Contributions, ideas, and improvements are welcome!
Feel free to open an issue or submit a PR.

