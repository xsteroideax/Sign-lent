import os
import json
import cv2
import shutil
from pytube import YouTube

def download_youtube_video(url, output_path):
    """
    Downloads a YouTube video given its URL.
    
    Parameters:
    - url: str, the URL of the YouTube video.
    - output_path: str, the path where the video will be saved.
    
    Returns:
    - bool, True if the download was successful, otherwise False.
    """
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        if stream:
            stream.download(output_path=output_path, filename="temp_video.mp4")
            return True
        else:
            print("No suitable video stream found.")
            return False
    except Exception as e:
        print(f"Error downloading video: {e}")
        return False

def video_to_frames(video_path, size=None):
    # (No changes needed)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return []

    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            if size:
                frame = cv2.resize(frame, size)
            frames.append(frame)
        else:
            break

    cap.release()
    return frames

def convert_frames_to_video(frame_array, path_out, size, fps=25):
    # (No changes needed)
    if not frame_array:
        print("No frames to write.")
        return

    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in frame_array:
        out.write(frame)
    out.release()

def extract_frame_as_video(src_video_path, start_frame, end_frame, output_path):
    # (No changes needed)
    frames = video_to_frames(src_video_path)
    
    if start_frame == 1 and end_frame == -1:
        start_frame = 0
        end_frame = len(frames) - 1
    else:
        end_frame = min(end_frame, len(frames) - 1)

    segment = frames[start_frame:end_frame + 1]
    if segment:
        height, width, layers = segment[0].shape
        size = (width, height)
        convert_frames_to_video(segment, output_path, size, fps=25)
    else:
        print("No frames extracted for the video segment.")

def process_dataset(json_path):
    # (Modified to include downloading videos if needed)
    with open(json_path, 'r') as f:
        content = json.load(f)
    
    output_dir = 'processed_videos'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for entry in content:
        instances = entry['instances']
        for inst in instances:
            video_id = inst['video_id']
            frame_start = inst['frame_start']
            frame_end = inst['frame_end']
            url = inst['url']
            
            # Define video paths
            src_video_path = os.path.join('raw_videos', f"{video_id}.mp4")
            output_video_path = os.path.join(output_dir, f"{video_id}_segment.mp4")

            # Check if the video exists locally; if not, attempt to download
            if not os.path.exists(src_video_path):
                print(f"Video {video_id} not found locally. Attempting to download...")
                os.makedirs('raw_videos', exist_ok=True)
                if download_youtube_video(url, 'raw_videos'):
                    # Rename the downloaded video to match the expected video_id
                    os.rename(os.path.join('raw_videos', 'temp_video.mp4'), src_video_path)
                else:
                    print(f"Failed to download video {video_id}. Skipping.")
                    continue

            # Process the video now that it is available locally
            print(f"Processing video {video_id} from frames {frame_start} to {frame_end}.")
            extract_frame_as_video(src_video_path, frame_start, frame_end, output_video_path)

if __name__ == "__main__":
    process_dataset('WLASL_v0.3.json')
