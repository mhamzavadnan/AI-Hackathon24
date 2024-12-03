import os
import cv2
import math
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
import numpy as np 
import albumentations as A
from albumentations.core.composition import ReplayCompose
from albumentations.pytorch import ToTensorV2

def split_videos_and_extract_features(
    parent_directory, video_dir, text_dir, output_video_dir, output_text_dir, feature_dir, 
    frame_window_size, overlap_size, use_augmentations=False
):
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_text_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    train_bundle_path = parent_directory + "/train.bundle"
    test_bundle_path = parent_directory + "/test.bundle"

    train_bundle = open(train_bundle_path, "w")
    test_bundle = open(test_bundle_path, "w")

    augmentation = None
    if use_augmentations:
        augmentation = ReplayCompose([
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.HorizontalFlip(p=0.5)
        ])

    resnet50 = models.resnet50(pretrained=True)
    resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
    resnet50.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet50.to(device)

    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    num_copies = 2    
    for video_file in video_files:
        for z in range(num_copies):
            video_path = os.path.join(video_dir, video_file)
            text_path = os.path.join(text_dir, os.path.splitext(video_file)[0] + ".txt")
            
            if not os.path.exists(text_path):
                print(f"Warning: No matching text file for {video_file}")
                continue
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

            with open(text_path, 'r') as f:
                labels = f.read().splitlines()

            if len(labels) != total_frames:
                print(f"Warning: Mismatch between number of frames and labels for {video_file}")
                continue
            
            for start_frame in range(0, total_frames - frame_window_size + 1, overlap_size):
                
                end_frame = start_frame + frame_window_size
                print(start_frame, "    ", end_frame)
                split_video_path = os.path.join(output_video_dir, f"{os.path.splitext(video_file)[0]}_part{start_frame+1}_to_{end_frame}_copy{num_copies}.mp4")
                split_text_path = os.path.join(output_text_dir, f"{os.path.splitext(video_file)[0]}_part{start_frame+1}_to_{end_frame}_copy{num_copies}.txt")
                split_feature_path = os.path.join(feature_dir, f"{os.path.splitext(video_file)[0]}_part{start_frame+1}_to_{end_frame}_copy{num_copies}.npy")

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(split_video_path, fourcc, frame_rate, 
                                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frames = []
                replay_params = None

                for i in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if use_augmentations:
                        if replay_params is None:
                            augmented = augmentation(image=frame_rgb)
                            replay_params = augmented['replay']
                        else:
                            augmented = ReplayCompose.replay(replay_params, image=frame_rgb)
                        frame_aug = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
                    else:
                        frame_aug = frame

                    out.write(frame_aug)
                    input_tensor = transform(frame_aug).unsqueeze(0).to(device)
                    frames.append(input_tensor)

                out.release()

                with open(split_text_path, 'w') as f:
                    f.write("\n".join(labels[start_frame:end_frame]))

                if frames:
                    inputs = torch.cat(frames, dim=0)
                    with torch.no_grad():
                        features = resnet50(inputs).squeeze(-1).squeeze(-1)
                    np.save(split_feature_path, features.cpu().numpy())

                if video_file.startswith("S1"):
                    test_bundle.write(split_text_path + "\n")
                else:
                    train_bundle.write(split_text_path + "\n")

            cap.release()

    train_bundle.close()
    test_bundle.close()
    print("Splitting and feature extraction with overlapping splits completed.")

parent_directory = "split60_overlap_album"
os.makedirs(parent_directory, exist_ok=True)

video_directory = "videos_all"
text_directory = "actions_all"
output_videos = parent_directory + "/videos"
output_texts = parent_directory + "/groundTruth"
feature_directory = parent_directory + "/features"
frame_window_size = 60
overlap_size = 5

split_videos_and_extract_features(
    parent_directory, video_directory, text_directory, output_videos, output_texts, feature_directory, 
    frame_window_size, overlap_size, use_augmentations=True
)
