import torch
import torch.nn as nn
import cv2
import numpy as np
from pytorch_i3d import InceptionI3d
 
# Initialize the model for 400 classes first
i3d = InceptionI3d(400, in_channels=3)
i3d.replace_logits(num_classes=2000)  # Replace the final logits layer to output 2000 classes
 
# Load the pre-trained weights
pretrained_weights = torch.load('C:\\Users\\Worawalan\\Downloads\\signlangage_copy\\code\\I3D\\saved_model.pth', map_location=torch.device('cpu'))
model_dict = i3d.state_dict()

# Filter out the final classification layer from the pretrained weights
pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_weights)
i3d.load_state_dict(model_dict)
 
i3d.eval()  # Set the model to evaluation mode
i3d = nn.DataParallel(i3d).cuda()
 
def run_video_inference(model, video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
   
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    q = 0
    while True:
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to capture frame.")
                break
           
            # Display the current frame
            cv2.imshow('Video Feed', frame)
 
            # Resize and normalize the frame for the model input
            frame_resized = cv2.resize(frame, (224, 224))  # Resize to 224x224 for the model input
            frame_normalized = (frame_resized / 255.) * 2 - 1  # Normalize to [-1, 1]
            frames.append(frame_normalized)
        q = q+1
        print("q=",q)
        if q >= num_frames:
            print("Not enough frames captured. Exiting...")
            break
 
        # Stack frames to create a sequence for the model input
        frames = np.stack(frames)  # Stack to create (num_frames, height, width, channels)
        frames = torch.Tensor(frames).permute(3, 0, 1, 2).unsqueeze(0).cuda()  # Rearrange to (batch, channels, time, height, width)
 
        with torch.no_grad():
            predictions = model(frames)  # Predictions shape: (batch, num_classes, num_frames)
            avg_predictions = torch.mean(predictions, dim=2)  # Average over frames
 
            # Apply softmax to get probabilities
            probabilities = torch.softmax(avg_predictions, dim=1)
 
            # Get the most probable class label
            out_labels = torch.argmax(probabilities, dim=1)  # Get the label for the batch
 
        print("Predicted Label:", out_labels.item())  # Display the predicted label
 
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    video_path = r"C:\Users\Worawalan\Downloads\signlangage_copy\raw_videos\50529.mp4"  # Update with the path to your video file
    run_video_inference(i3d, video_path)
 