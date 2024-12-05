import os
import cv2
import argparse

def overlay_action_labels(video_dir, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if f.endswith('.avi')]

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        label_file = os.path.splitext(video_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            print(f"No label file found for {video_file}. Skipping...")
            continue

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = os.path.join(output_dir, video_file)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

        with open(label_path, 'r') as f:
            labels = f.readlines()

        frame_actions = []
        for line in labels:
            start, end, action, _ = line.strip().split(',')
            frame_actions.append((int(start), int(end), action))

        current_action = None
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            for start, end, action in frame_actions:
                if start <= frame_idx <= end:
                    current_action = action
                    break
                else:
                    current_action = None

            if current_action:
                text = f"Action: {current_action}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            out.write(frame)
            frame_idx += 1

            if frame_idx >= total_frames:
                break

        cap.release()
        out.release()
        print(f"Processed {video_file} and saved to {output_path}")

    print("All videos processed and labeled.")

def get_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Overlay action labels on videos.")
    parser.add_argument(
        "video_dir", type=str, help="Path to the directory containing video files"
    )
    parser.add_argument(
        "label_dir", type=str, help="Path to the directory containing label files"
    )
    parser.add_argument(
        "output_dir", type=str, help="Path to the output directory to save labeled videos"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()

    overlay_action_labels(args.video_dir, args.label_dir, args.output_dir)
