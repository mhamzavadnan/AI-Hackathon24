import os
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description="Extract video filenames and create all_files.txt")
    parser.add_argument("video_folder_path", type=str, help="Path to the folder containing video files")
    parser.add_argument("output_file", type=str, help="Path to save the generated all_files.txt")
    return parser.parse_args()

def main():
    args = get_arguments()

    video_files = [f.split('.')[0] for f in os.listdir(args.video_folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
    with open(args.output_file, 'w') as f:
        for video in video_files:
            f.write(video + '\n')

    print(f"{args.output_file} has been created.")

if __name__ == "__main__":
    main()
