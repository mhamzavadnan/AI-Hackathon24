import os
import cv2
import argparse

def overlay_action_labels(frame_dir, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files from the frame directory (e.g., .jpg or .png)
    frame_files = [f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))]
    frame_files.sort()  # Sorting to ensure the frames are processed in correct order

    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        label_file = os.path.splitext(frame_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            print(f"No label file found for {frame_file}. Skipping...")
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Failed to read {frame_file}. Skipping...")
            continue

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        # Prepare output video writer (if you want to save the results in a video)
        output_path = os.path.join(output_dir, frame_file)
        
        with open(label_path, 'r') as f:
            labels = f.readlines()

        frame_actions = []
        for line in labels:
            start, end, action, _ = line.strip().split(',')
            frame_actions.append((int(start), int(end), action))

        # Assuming that the frame filename is the frame number
        frame_idx = int(os.path.splitext(frame_file)[0])

        # Find the corresponding action for this frame
        current_action = None
        for start, end, action in frame_actions:
            if start <= frame_idx <= end:
                current_action = action
                break

        # Overlay action label on the frame
        if current_action:
            text = f"Action: {current_action}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save the frame with overlaid text
        cv2.imwrite(output_path, frame)

        print(f"Processed {frame_file} with action: {current_action if current_action else 'None'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay action labels on frames")
    parser.add_argument("frame_dir", help="Directory containing frames (images)")
    parser.add_argument("label_dir", help="Directory containing label files")
    parser.add_argument("output_dir", help="Directory to save the processed frames")

    args = parser.parse_args()

    overlay_action_labels(args.frame_dir, args.label_dir, args.output_dir)
