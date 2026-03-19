import os
import time
import random
import argparse
import csv
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from google import genai
from google.genai import types
from google.genai.errors import ClientError
from tqdm import tqdm

prompt = """
You are a perceptive visual assistant. Given a video input, filter out videos that do not meet drone footage standards. Return one or more of the following tags if applicable:

* [Not drone] — if the video is not drone footage (e.g., people interviews, selfie videos, handheld phone recordings, videos recorded through airplane windows, or time-lapse videos recorded with a phone).
* [Multiple scenes] — if the video contains dissolve transitions or similar editing effects that indicate multiple distinct scenes stitched together.
* [Watermark] — if any watermark, logo, or text overlay is visible in the video.
* [Low quality] — if the video suffers from low resolution, excessive compression artifacts, severe motion blur, poor lighting, or other visual issues that obscure scene understanding.
* [Abrupt motion] — if the camera exhibits sudden, disorienting changes in movement or orientation (e.g., rapid vertical flips, erratic directional jumps, or unnatural acceleration) that break continuity. Smooth accelerations or sharp turns are acceptable.

If any of these tags apply, return only the applicable tag(s).

Output only the final tags, with no extra commentary. If no tags are returned, return "Valid".
"""

api_keys = ['YOUR_GEMINI_API_KEY']

def filter_video(video_path, max_retries=5):
    try:
        client = genai.Client(
            api_key=random.choice(api_keys),
        )
    except Exception as e:
        error_msg = f"Could not initialize client: {str(e)}"
        logging.error(error_msg)
        return f"ERROR: {error_msg}"

    uploaded_file = None
    try:
        logging.info(f"Uploading file: {os.path.basename(video_path)}...")
        uploaded_file = client.files.upload(file=video_path)
    except Exception as e:
        error_msg = f"Failed to upload file {os.path.basename(video_path)}: {str(e)}"
        logging.error(error_msg)
        return f"ERROR: {error_msg}"

    logging.info(f"File '{uploaded_file.name}' uploaded. Waiting for it to become ACTIVE...")
    timeout_seconds = 300
    start_time = time.time()
    while True:
        try:
            file_state = client.files.get(name=uploaded_file.name)
            
            if file_state.state.name == "ACTIVE":
                logging.info(f"File '{uploaded_file.name}' is now ACTIVE.")
                break
            elif file_state.state.name == "FAILED":
                error_msg = f"File processing failed on the server for {os.path.basename(video_path)}."
                logging.error(error_msg)
                try: client.files.delete(name=uploaded_file.name)
                except: pass
                return f"ERROR: {error_msg}"
            
            if time.time() - start_time > timeout_seconds:
                error_msg = f"Timeout waiting for file {os.path.basename(video_path)} to become ACTIVE."
                logging.error(error_msg)
                try: client.files.delete(name=uploaded_file.name)
                except: pass
                return f"ERROR: {error_msg}"
            
            logging.info(f"File state is {file_state.state.name}, waiting...")
            time.sleep(10)

        except Exception as e:
            error_msg = f"Could not get file state for {uploaded_file.name}. Reason: {str(e)}"
            logging.error(error_msg)
            return f"ERROR: {error_msg}"

    files = [uploaded_file]
    model_name = "gemini-2.5-flash"
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    tags = ""
    retries = 0

    try:
        while retries < max_retries:
            try:
                logging.info(f"Generating content for {os.path.basename(video_path)}...")
                for chunk in client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if chunk.text is not None:
                        tags += chunk.text
                break
            except ClientError as e:
                if hasattr(e, 'code') and e.code == 429:
                    retries += 1
                    if retries >= max_retries:
                        error_msg = f"API quota exceeded after {max_retries} retries."
                        logging.error(error_msg)
                        return f"ERROR: {error_msg}"
                    wait_time = (2 ** retries) + random.random()
                    logging.warning(f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    error_msg = f"ClientError: {str(e)}"
                    logging.error(error_msg)
                    return f"ERROR: {error_msg}"
            except Exception as ex:
                error_msg = f"Unexpected exception: {str(ex)}"
                logging.error(error_msg)
                return f"ERROR: {error_msg}"
    finally:
        if uploaded_file:
            try:
                logging.info(f"Deleting file {uploaded_file.name}...")
                client.files.delete(name=uploaded_file.name)
            except Exception as e:
                logging.warning(f"Failed to delete file {uploaded_file.name}: {str(e)}")

    return tags


def process_videos_multiprocess(directory_path, output_file="filters.csv", max_workers=4):
    logging.info(f"Scanning for videos in '{directory_path}'...")
    video_files = []
    VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv')
    try:
        for entry in os.scandir(directory_path):
            if entry.is_file() and entry.name.lower().endswith(VIDEO_EXTENSIONS):
                video_files.append(os.path.abspath(entry.path))
        if not video_files:
            logging.info(f"No video files found in '{directory_path}'. Exiting.")
            return
        logging.info(f"Found {len(video_files)} total video files.")
    except FileNotFoundError:
        logging.error(f"Directory not found at '{directory_path}'. Exiting.")
        return

    results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if len(row) == 2 and row[0]:
                        results[os.path.abspath(row[0])] = row[1]
        except (IOError, StopIteration, IndexError) as e:
            logging.warning(f"Could not read existing results from {output_file}. Starting fresh. Error: {e}")
            results = {}

    remaining_videos = [vid for vid in video_files if vid not in results]
    logging.info(f"{len(results)} videos are already processed; {len(remaining_videos)} remain.")
    
    if not remaining_videos:
        logging.info("All videos are already processed. Nothing to do.")
        return

    logging.info(f"Starting video processing with up to {max_workers} worker processes.")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {executor.submit(filter_video, vid): vid for vid in remaining_videos}

        with tqdm(total=len(remaining_videos), desc="Processing Videos", unit="vid", dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    tags = future.result()
                except Exception as e:
                    tags = f"ERROR: Future completed with an exception: {str(e)}"
                    logging.error(f"Error processing {os.path.basename(video_path)}: {tags}")

                results[video_path] = tags
                pbar.update(1)
                pbar.set_postfix({"Last": os.path.basename(video_path)})

                try:
                    with open(output_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['video_path', 'tags'])
                        writer.writerows(results.items())
                except IOError as e:
                    logging.warning(f"Failed to write intermediate results to {output_file}. Error: {e}")

    logging.info(f"All videos processed. Final results saved to {output_file}.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(processName)s] - [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler("filter.log"),
            logging.StreamHandler()
        ]
    )

    noisy_loggers = [
        "google_genai",
        "httpx",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.info("Logger configuration complete. All noisy library logs are set to WARNING level.")


    parser = argparse.ArgumentParser(description="Scan a directory for videos, filter out videos that do not meet drone footage standards using the Gemini API, and save to a CSV file.")
    
    parser.add_argument("--directory", default='./data', help="Directory containing video files to process.")
    parser.add_argument("--output", "-o", default="filters.csv", help="Output CSV file for filters (default: filters.csv).")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel processes to use (default: 4).")
    
    args = parser.parse_args()
    
    process_videos_multiprocess(
        directory_path=args.directory,
        output_file=args.output,
        max_workers=args.workers
    )