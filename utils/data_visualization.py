import cv2
import numpy as np
import json
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt

# Helper function to convert RLE to polygons
def rle_to_polygon(segmentation, width, height):
    if isinstance(segmentation, list):  # Polygon format (list of points)
        polygons = []
        for poly in segmentation:
            polygons.append(np.array(poly).reshape(-1, 2))  # Handle each polygon as a list of points
        return polygons
    elif isinstance(segmentation, dict):  # RLE format (dict)
        rle = coco_mask.frPyObjects(segmentation, height, width)
        mask = coco_mask.decode(rle)  # Decode the RLE to a binary mask
       
        # Debugging: Check if mask is valid (non-zero)
        if np.count_nonzero(mask) == 0:
            print(f"Warning: Decoded mask is empty for segmentation {segmentation}")
        return mask_to_polygons(mask)  # Convert the binary mask to polygons
    else:
        raise ValueError(f"Unsupported segmentation format: {type(segmentation)}")

# Convert binary mask to polygon coordinates
def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) >= 3:  # Ignore small contours that are not valid polygons
            polygons.append(cnt.reshape(-1, 2))
    return polygons

# Draw bounding boxes, segmentations, and keypoints
def visualize_annotations(image_path, annotations, output_image_path, visualize_bboxes=True, visualize_keypoints=False, visualize_segmentations=True):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Unable to load image {image_path}. Skipping this image.")
        return  # Skip further processing for this image

    # Get image dimensions
    height, width, _ = img.shape

    # Process bounding boxes
    if visualize_bboxes:
        for ann in annotations:
            bbox = ann.get('bbox', None)  # [x, y, w, h]
            if bbox:
                x, y, w, h = map(int, bbox)
                # Draw the bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Process segmentation if it exists and visualize
            segmentation = ann.get('segmentation', None)
            if segmentation and visualize_segmentations:
                polygons = rle_to_polygon(segmentation, width, height)
                if polygons:
                    for polygon in polygons:
                        polygon = np.array(polygon).reshape(-1, 2).astype(np.int32)
                        # Draw the segmentation polygons
                        cv2.polylines(img, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)

    # Process keypoints if needed
    if visualize_keypoints:
        for ann in annotations:
            keypoints = ann.get('keypoints', [])  # List of (x, y, visibility)
            for i in range(0, len(keypoints), 3):
                x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
                if v == 2:  # If keypoint is visible
                    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red for visible keypoints

    # Save the processed image
    cv2.imwrite(output_image_path, img)
    print(f"Processed image saved at: {output_image_path}")

    # Optionally display the image using matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# Main function to process all images
def process_images(annotation_file, image_folder, output_folder):
    # Load annotations from the file
    with open(annotation_file, 'r') as f:
        annotations_data = json.load(f)

    # Ask user for what to visualize
    print("What would you like to visualize?")
    print("1. Bounding boxes only")
    print("2. Bounding boxes and keypoints")
    print("3. Bounding boxes, segmentations, and keypoints")
   
    choice = input("Enter your choice (1, 2, or 3): ").strip()

    # Set visualization flags based on user input
    if choice == "1":
        visualize_bboxes = True
        visualize_keypoints = False
        visualize_segmentations = False
    elif choice == "2":
        visualize_bboxes = True
        visualize_keypoints = True
        visualize_segmentations = False
    elif choice == "3":
        visualize_bboxes = True
        visualize_keypoints = True
        visualize_segmentations = True
    else:
        print("Invalid choice. Defaulting to bounding boxes and keypoints.")
        visualize_bboxes = True
        visualize_keypoints = True
        visualize_segmentations = False

    # Process each image and corresponding annotations
    for image_info in annotations_data['images']:
        image_id = image_info['id']
        image_filename = image_info['file_name']
        image_path = f"{image_folder}/{image_filename}"

        # Get the corresponding annotations for this image (bbox, keypoints, and segmentation)
        image_annotations = [ann for ann in annotations_data['annotations'] if ann['image_id'] == image_id]

        # Create the output image path
        output_image_path = f"{output_folder}/{image_filename}"

        # Visualize annotations on the image
        visualize_annotations(image_path, image_annotations, output_image_path, visualize_bboxes, visualize_keypoints, visualize_segmentations)

# Example Usage
image_folder = "train2017"  # The directory containing your image files
annotation_file = "annotations/instances_train2017.json"  # This file contains bbox, keypoints, and segmentation
output_folder = "output"  # The directory where you want to save processed images

process_images(annotation_file, image_folder, output_folder)
