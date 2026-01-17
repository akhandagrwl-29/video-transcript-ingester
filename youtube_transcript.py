from pytube import Playlist
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import os
import ssl
import certifi
import time
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig


ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


def fetch_playlist_transcripts(playlist_url: str, output_dir="transcripts"):
    playlist = Playlist(playlist_url)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(playlist.video_urls)} videos in playlist")

    for idx, video_url in enumerate(playlist.video_urls, start=1):
        video_id = video_url.split("v=")[-1]
        time.sleep(1)

        try:
            proxy_username = os.environ.get("PROXY_USERNAME")
            proxy_password = os.environ.get("PROXY_PASSWORD")
            
            ytt_api = YouTubeTranscriptApi(
                proxy_config=WebshareProxyConfig(
                    proxy_username=proxy_username,
                    proxy_password=proxy_password,
                )
            )
            transcript_list = ytt_api.fetch(video_id)

            output_file = os.path.join(output_dir, f"{video_id}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                for snippet in transcript_list:
                    # print(snippet.text)
                    f.write(snippet.text + " ")

            print(f"[{idx}] Saved transcript → {output_file}")

        except TranscriptsDisabled:
            print(f"[{idx}] Transcripts disabled for video: {video_id}")
        except NoTranscriptFound:
            print(f"[{idx}] No transcript found for video: {video_id}")
        except Exception as e:
            print(f"[{idx}] Failed for {video_id}: {e}")


if __name__ == "__main__":
    playlist_url = "https://youtube.com/playlist?list=PLMCXHnjXnTnvo6alSjVkgxV-VH6EPyvoX"
    fetch_playlist_transcripts(playlist_url)
