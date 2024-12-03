import cv2
import numpy as np
import json

# Draw bounding boxes and keypoints on the image
def visualize_annotations(image, annotations, visualize_bboxes=True, visualize_keypoints=True):
    # Process bounding boxes
    if visualize_bboxes:
        for ann in annotations:
            bbox = ann.get('bbox', None)  # [x, y, w, h]
            if bbox:
                x, y, w, h = map(int, bbox)
                # Draw the bounding box (rectangle)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Process keypoints
    if visualize_keypoints:
        for ann in annotations:
            keypoints = ann.get('keypoints', None)  # [x, y, visibility, ...]
            if keypoints:
                for i in range(0, len(keypoints), 3):  # Loop through keypoints
                    x, y, visibility = keypoints[i], keypoints[i+1], keypoints[i+2]
                    if visibility > 0:  # Only draw keypoints that are visible
                        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)  # Draw a red circle

    return image

# Main function to process video and annotations
def process_video(video_path, annotation_path, output_video_path=None):
    # Load the annotations file
    with open(annotation_path, 'r') as f:
        annotations_data = json.load(f)  # This should be a dictionary with the "annotations" key

    # Ensure we are working with the correct structure
    if 'annotations' not in annotations_data:
        print("Error: The annotations data does not contain the expected 'annotations' key.")
        return

    annotations = annotations_data['annotations']  # List of annotations

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter to save the output video if needed
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process each frame in the video
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break  # Break if the frame could not be read

        # Get the current annotations for the frame (based on image_id)
        current_annotations = [ann for ann in annotations if ann['image_id'] == frame_idx]

        # Visualize annotations on the current frame
        frame_with_annotations = visualize_annotations(frame, current_annotations)

        # Write the frame to the output video if needed
        if output_video_path:
            out.write(frame_with_annotations)

        # Display the frame with annotations
        cv2.imshow("Frame", frame_with_annotations)

        # Exit on key press (Esc)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Clean up
    cap.release()
    if output_video_path:
        out.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'amur__1_moded.mp4'
annotation_path = 'annotations_coco.json'
output_video_path = 'output_video_with_annotations.mp4'

process_video(video_path, annotation_path, output_video_path)
