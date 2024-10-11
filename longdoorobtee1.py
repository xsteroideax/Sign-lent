import torch
import torch.nn as nn
import cv2
import numpy as np
from pytorch_i3d import InceptionI3d
import pandas as pd


# Initialize the model for 400 classes first
i3d = InceptionI3d(400, in_channels=3)
i3d.replace_logits(num_classes=2000)  # Replace the final logits layer to output 2000 classes

# Load the pre-trained weights
pretrained_weights = torch.load('C:\\Users\\Worawalan\\Downloads\\signlangage_copy\\code\\I3D\\saved_model.pth', map_location=torch.device('cpu'))
model_dict = i3d.state_dict()

label_names_df = pd.read_csv(r"C:\Users\Worawalan\Downloads\label_name_jingjing - wlasl_class_list.csv")

# Filter out the final classification layer from the pretrained weights
pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_weights)
i3d.load_state_dict(model_dict)

i3d.eval()  # Set the model to evaluation mode
i3d = nn.DataParallel(i3d).cuda()

def run_camera_inference(model, cam_index=0, num_frames=9,label_names_df=label_names_df):
    cap = cv2.VideoCapture(cam_index)
  
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera.")
                break
            
            # Display the original camera feed
            cv2.imshow('Camera Feed', frame)

            # Resize and normalize the frame for the model input
            frame_resized = cv2.resize(frame, (224, 224))  # Resize to 224x224 for the model input
            frame_normalized = (frame_resized / 255.) * 2 - 1  # Normalize to [-1, 1]
            frames.append(frame_normalized)

        if len(frames) < num_frames:
            print("Not enough frames captured.")
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

        if out_labels.item() < len(label_names_df['meaning']):
            #predicted_label_name = label_names_df[meaning].iloc[index, 0]  # Adjust the column index if needed
            predicted_label_name = label_names_df.loc[(label_names_df['index']==out_labels.item()),['meaning']]
            print("Predicted Label:", predicted_label_name)  # Display the predicted label name
        else:
            print("Predicted Label: Index out of range")  # Handle cases where the index is out of bounds

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    run_camera_inference(i3d)
    

