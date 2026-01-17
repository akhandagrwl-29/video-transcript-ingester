from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi


def get_video_id(youtube_url: str) -> str:
    parsed_url = urlparse(youtube_url)

    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        return parse_qs(parsed_url.query).get("v", [None])[0]

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    raise ValueError("Invalid YouTube URL")


def fetch_transcript(youtube_url: str):
    video_id = get_video_id(youtube_url)

    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id)

    output_file = f"{video_id}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for snippet in transcript_list:
            f.write(snippet.text + " ")

    print(f"Transcript saved to {output_file}")


if __name__ == "__main__":
    url = "https://youtu.be/zg_66Q3oSss"
    fetch_transcript(url)
