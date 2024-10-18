import math
import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import videotransforms
import numpy as np
import torch.nn.functional as F
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset_all import NSLT as Dataset
import cv2

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set CUDA devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
args = parser.parse_args()

def save_model(model, filepath):
    """Saves the model to the specified filepath."""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")
    
# Function to load RGB frames from video
def load_rgb_frames_from_video(video_path, start=0, num=-1):
    vidcap = cv2.VideoCapture(video_path)
    frames = []

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    if num == -1:
        num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for offset in range(num):
        success, img = vidcap.read()
        if not success:
            break
        w, h, c = img.shape
        sc = 224 / w  # Resize ratio
        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.0) * 2 - 1  # Normalize to [-1, 1]
        frames.append(img)

    return torch.Tensor(np.asarray(frames, dtype=np.float32))

# Function to run the model training and evaluation
def run(init_lr=0.1,
        max_steps=64000,
        mode='rgb',
        root=r"C:\Users\Asus\OneDrive\kosen2\signlent\preprocess2.py",
        train_split=r"C:\Users\Asus\OneDrive\kosen2\signlent\WLASL_v0.3.json",
        batch_size=45,
        save_model='',
        weights=None):
    
    # Transform for validation dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    # Load model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load(r"C:\Users\Asus\OneDrive\kosen2\signlent\weights\flow_imagenet.pt", map_location=torch.device('cpu'), weights_only=True))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load(r"C:\Users\Asus\OneDrive\kosen2\signlent\weights\rgb_imagenet.pt", map_location=torch.device('cpu'), weights_only=True))

    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights, map_location=torch.device('cpu'), weights_only=True))
    i3d.cpu()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    # Initialize metrics
    correct = 0
    correct_5 = 0
    correct_10 = 0
    top1_fp = np.zeros(num_classes, dtype=np.int32)
    top1_tp = np.ones(num_classes, dtype=np.int32)
    top5_tp = np.ones(num_classes, dtype=np.int32)
    top5_fp = np.zeros(num_classes, dtype=np.int32) 
    top10_fp = np.zeros(num_classes, dtype=np.int32)
    top10_tp = np.ones(num_classes, dtype=np.int32)

    # Evaluation loop
    for data in val_dataloader:
        inputs, labels, video_id = data  
        per_frame_logits = i3d(inputs)

        predictions = torch.max(per_frame_logits, dim=2)[0]
        out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
        predicted_label = torch.argmax(predictions[0]).item() 

        if labels[0].item() in out_labels[-5:]:
            correct_5 += 1
            top5_tp[labels[0].item()] += 1
        else:
            top5_fp[labels[0].item()] += 1
        if labels[0].item() in out_labels[-10:]:
            correct_10 += 1
            top10_tp[labels[0].item()] += 1
        else:
            top10_fp[labels[0].item()] += 1
        if predicted_label == labels[0].item():
            correct += 1
            top1_tp[labels[0].item()] += 1
        else:
            top1_fp[labels[0].item()] += 1

        print(video_id, float(correct) / len(val_dataloader), float(correct_5) / len(val_dataloader), float(correct_10) / len(val_dataloader))

    # Calculate top-k average per class accuracy
    top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp))
    top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp))
    top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp))
    print('top-k average per class acc: {}, {}, {}'.format(top1_per_class, top5_per_class, top10_per_class))

if __name__ == '__main__':
    i3d = InceptionI3d(400, in_channels=3)
    mode = 'rgb'
    num_classes = 2000
    save_model = 'code/'
    root = r"C:\Users\Asus\OneDrive\kosen2\signlent\preprocess2.py"
    train_split = 'preprocess/nslt_{}.json'.format(num_classes)
    weights = r"C:\Users\Asus\OneDrive\kosen2\signlent\FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt"
    
    # Run the training/evaluation
    run(mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights)
    
    # Save the model state
    torch.save(i3d.state_dict(), r"C:\Users\Asus\OneDrive\kosen2\signlent\saved_model.pth")
