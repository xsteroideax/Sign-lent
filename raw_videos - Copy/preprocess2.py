import os
import cv2
 
# Path to the directory
directory_path = r"C:\Users\Asus\OneDrive\kosen2\signlent\processed_videos"
 
# List all files in the directory
contents = os.listdir(directory_path)
 
# Loop to find a video file
video_file = None
for file_name in contents:
    if file_name.endswith(('.mp4', '.avi', '.mov')):  # Add other formats if needed
        video_file = os.path.join(directory_path, file_name)
        break
 
# Check if a video file was found
if video_file is None:
    print("No video file found in the directory.")
else:
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        frames = []
        size = (224, 224)  # Specify size if you want to resize frames
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
        print(f"Frames per second: {fps}")
 
        # Capture every nth frame
        n = 1  # Adjust n to capture every nth frame (e.g., 2 for every second frame)
        frame_count = 0
 
        while True:
            ret, frame = cap.read()
 
            if ret:
                if frame_count % n == 0:  # Capture every nth frame
                    if size:
                        frame = cv2.resize(frame, size)  # Resize if needed
                    frames.append(frame)
                frame_count += 1
            else:
                print("Finished reading the video.")
                break
 
        # Release the video capture object
        cap.release()
 
        # Optionally, you can now process the frames, save them, or visualize them
        print(f"Total frames captured: {len(frames)}")