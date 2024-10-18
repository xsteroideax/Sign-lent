import torch
import torch.nn as nn
import cv2
import numpy as np
from pytorch_i3d import InceptionI3d
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pandas as pd
import pyttsx3
from googletrans import Translator

# Load label names from CSV
label_names_df = pd.read_csv(r"C:\Users\Asus\OneDrive\kosen2\signlent\label_name_jingjing.csv")

# Initialize the I3D model
def initialize_model():
    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(num_classes=2000)

    pretrained_weights = torch.load(
        r"C:\Users\Asus\OneDrive\kosen2\signlent\saved_model.pth",
        map_location=torch.device('cpu'),
        weights_only=True
    )
    model_dict = model.state_dict()
    pretrained_weights = {k: v for k, v in pretrained_weights.items()
                          if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_weights)
    model.load_state_dict(model_dict)
    model.eval()
    
    return nn.DataParallel(model).cuda() if torch.cuda.is_available() else model

# Preprocess frames to normalize them
def preprocess_frame(frame, size=(224, 224)):
    frame_resized = cv2.resize(frame, size)
    frame_normalized = (frame_resized / 255.0) * 2 - 1
    return frame_normalized

# Global variables for video state management
cap = None  
running = False  
frames = []  
predicted_label_name = ""

def run_video_inference(model, video_path, num_frames=9, display_delay=500):
    global cap, running, frames, predicted_label_name 

    # Reset frames for new video
    frames = []

    if cap is not None:
        cap.release()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        output_label.config(text="Error: Could not open video.")
        return

    output_label.config(text="Processing...")  # Indicate processing
    running = True  

    def process_frames():
        global running, predicted_label_name

        if not running:
            return  

        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to capture frame.")
            cap.release()
            running = False
            return

        # Preprocess and append frame for inference
        frames.append(preprocess_frame(frame))

        if len(frames) == num_frames:
            frames_tensor = torch.Tensor(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
            if torch.cuda.is_available():
                frames_tensor = frames_tensor.cuda()

            with torch.no_grad():
                predictions = model(frames_tensor)
                avg_predictions = torch.mean(predictions, dim=2)
                probabilities = torch.softmax(avg_predictions, dim=1)
                predicted_label = torch.argmax(probabilities, dim=1).item()
                print(f"Predicted Label: {predicted_label}, Probabilities: {probabilities}")  # Debugging line

            if predicted_label < len(label_names_df['meaning']):
                predicted_label_name = label_names_df.loc[label_names_df['index'] == predicted_label, 'meaning'].values[0]
                output_label.config(text=f"Prediction: {predicted_label_name}")
            else:
                output_label.config(text="Predicted Label: Index out of range")

            frames.clear()  # Clear frames after processing

        # Process the next frame immediately
        root.after(1, process_frames)

    def update_display():
        if not running:
            return

        # Show the current frame in the display area
        ret, frame = cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (224, 224))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
            video_label.config(image=img)
            video_label.image = img  # Keep a reference to avoid garbage collection

        # Schedule the next display update after the specified delay
        root.after(display_delay, update_display)

    process_frames()  # Start processing frames immediately
    update_display()  # Start updating display with delay

# Select a new video file
def select_video_path():
    global running, cap

    video_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=(("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*"))
    )

    if video_path:
        print(f"Selected video path: {video_path}")

        # Reset the output label and image before starting new inference
        if running:
            cap.release()  # Release the current video capture
            running = False  # Stop the current processing

        # Reset UI elements
        output_label.config(text="Prediction: ")  # Reset the output label
        video_label.config(image='')  # Clear the video label image

        run_video_inference(i3d, video_path)

def translate():
    global predicted_label_name
    translator = Translator()

    # Text to be translated
    text = predicted_label_name  

    # Create a new window for language selection
    root2 = tk.Tk()
    root2.title("Select Language")
    root2.geometry('450x290')

    # Canvas for layout
    canvas2 = tk.Canvas(root2, width=450, height=290)
    canvas2.pack()

    # Label for translated output
    output_label = tk.Label(canvas2, text="Translation: ", font=("Arial", 16))
    canvas2.create_window(150, 150, window=output_label)

    # Function to translate text based on the selected language
    def translates(lang):
        translated = translator.translate(text, dest=lang)
        translated_text = translated.text

        # Display the translated text
        output_label.config(text=f"{predicted_label_name}: {translated_text}")

    # Function to close the language selection window
    def exit_app():
        root2.destroy()

    # Buttons for language selection
    button_thai = tk.Button(root2, text="Thai", command=lambda: translates('th'), width=20, height=2)
    button_thai.place(x=0, y=250)

    button_japan = tk.Button(root2, text="Japanese", command=lambda: translates('ja'), width=20, height=2)
    button_japan.place(x=150, y=250)

    # Exit button
    button_exit = tk.Button(root2, text="Exit", command=exit_app, width=20, height=2)
    button_exit.place(x=300, y=250)

def read():
    global predicted_label_name
    print(predicted_label_name)
    # Initialize the TTS engine
    engine = pyttsx3.init()
    
    # Text to be converted to speech
    text = predicted_label_name
    
    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
    
    # Convert text to speech
    engine.say(text)
    
    # Wait for the speaking to finish
    engine.runAndWait()

# Initialize the I3D model
i3d = initialize_model()

# Create Tkinter window
root = tk.Toplevel()
root.title("Sign Language Recognition")
root.geometry('600x450')
 
# Create a canvas to hold video display and buttons
canvas = tk.Canvas(root, width=600, height=450)
canvas.pack()
 
# Video display area (fixed size of 224x224)
video_label = tk.Label(canvas, width=400, height=300)
canvas.create_window(300, 150, window=video_label)  # Move video label higher
 
# Prediction output label
output_label = tk.Label(canvas, text="Prediction: ", font=("Arial", 16))
canvas.create_window(300, 330, window=output_label)
 
# Button to select a video file
button = tk.Button(canvas, text="Select Video File", command=select_video_path)
canvas.create_window(300, 300, window=button)
 
button1 = tk.Button(canvas, text="Translate", command=translate, width=40, height=3)
canvas.create_window(150, 400, window=button1)
 
button2 = tk.Button(canvas, text="Read text", command=read, width=40, height=3)
canvas.create_window(450, 400, window=button2)
 
# Start the GUI loop
root.mainloop()
