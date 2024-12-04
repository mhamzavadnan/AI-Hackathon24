import cv2
import numpy as np
import json
import argparse
import os

FINGER_CONNECTIONS = {
    'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
    'index': [(5, 6), (6, 7), (7, 8)],
    'middle': [(9, 10), (10, 11), (11, 12)],
    'ring': [(13, 14), (14, 15), (15, 16)],
    'pinky': [(17, 18), (18, 19), (19, 20)]
}

def visualize_annotations(image, annotations, visualize_bboxes=True, visualize_keypoints=True):
    if visualize_bboxes:
        for ann in annotations:
            bbox = ann.get('bbox', None)
            if bbox:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if visualize_keypoints:
        for ann in annotations:
            keypoints = ann.get('keypoints', None)
            if keypoints:
                for i in range(0, len(keypoints), 3):
                    x, y, visibility = keypoints[i], keypoints[i+1], keypoints[i+2]
                    if visibility > 0:
                        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
                
                for finger, connections in FINGER_CONNECTIONS.items():
                    for (start, end) in connections:
                        start_x, start_y = int(keypoints[start * 3]), int(keypoints[start * 3 + 1])
                        end_x, end_y = int(keypoints[end * 3]), int(keypoints[end * 3 + 1])

                        if keypoints[start * 3 + 2] > 0 and keypoints[end * 3 + 2] > 0:
                            cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)

    return image

def process_video(video_path, annotation_path, output_video_name):
    with open(annotation_path, 'r') as f:
        annotations_data = json.load(f)

    if 'annotations' not in annotations_data:
        print("Error: The annotations data does not contain the expected 'annotations' key.")
        return

    annotations = annotations_data['annotations']

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        frame_annotations = [ann for ann in annotations if ann['image_id'] == frame_idx]

        frame_with_annotations = visualize_annotations(frame, frame_annotations)

        output_video.write(frame_with_annotations)

        cv2.imshow('Video with Annotations', frame_with_annotations)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hand annotations on a video")
    parser.add_argument('video_path', type=str, help="Path to the input video")
    parser.add_argument('annotation_path', type=str, help="Path to the annotation file (JSON)")
    parser.add_argument('output_video_name', type=str, help="Name for the output video")
    args = parser.parse_args()

    process_video(args.video_path, args.annotation_path, args.output_video_name)
