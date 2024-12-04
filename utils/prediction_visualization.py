import argparse
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="convert pred and gt list to images.")
    parser.add_argument(
        "input_dir",
        type=str,
        help="path to a files you want to convert (predictions)",
    )
    parser.add_argument(
        "action_dict_path",
        type=str,
        help="path to a action dict file",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="path to output img",
        default="output"
    )
    parser.add_argument(
        "--sliding_windows",
        type=int,
        help="sliding windows size",
        default=120
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        help="path to the ground truth files",
        required=True
    )
    return parser.parse_args()


def convert_arr2img(file_path, palette, actions_dict):
    with open(file_path, 'r') as file_ptr:
        list = file_ptr.read().split('\n')[:-1]
    
    array = np.array([actions_dict[name] for name in list])

    arr = array.astype(np.uint8)
    arr = np.tile(arr, (100, 1))
    
    return arr


def make_palette(num_classes):
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette


def main() -> None:
    args = get_arguments()
    action_dict_path = args.action_dict_path

    with open(action_dict_path, 'r') as file_ptr:
        actions = file_ptr.read().split('\n')[:-1]
    actions_dict = {a.split()[1]: int(a.split()[0]) for a in actions}

    palette = make_palette(len(actions_dict))

    os.makedirs(args.output_dir, exist_ok=True)

    pred_filenames = [f.strip() for f in os.listdir(args.input_dir)]
    gt_filenames = [f.strip() for f in os.listdir(args.ground_truth_dir)]

    vid_list = []
    for pred in pred_filenames:
        if pred.endswith('.txt'):
            pred_vid = os.path.splitext(pred)[0]
            for gt in gt_filenames:
                if gt.endswith('.txt'):
                    gt_vid = os.path.splitext(gt)[0]
                    if pred_vid == gt_vid:
                        vid_list.append(pred_vid)

    for vid in tqdm(vid_list, desc='Processing videos'):
        gt_file_path = os.path.join(args.ground_truth_dir, vid + '.txt')
        pred_file_path = os.path.join(args.input_dir, vid + '.txt')

        if not os.path.exists(gt_file_path):
            continue
        if not os.path.exists(pred_file_path):
            continue

        gt_arr = convert_arr2img(gt_file_path, palette, actions_dict)
        pred_arr = convert_arr2img(pred_file_path, palette, actions_dict)

        if gt_arr.shape[1] != pred_arr.shape[1]:
            gt_arr = gt_arr[:, :-1]
        
        arr = np.concatenate([gt_arr, pred_arr], axis=0)
        img = Image.fromarray(arr)
        img = img.convert("P")
        img.putpalette(palette)

        plt.figure()
        plt.title('GroundTruth vs Prediction')
        plt.imshow(img)
        plt.xlabel('Prediction')
        plt.gca().xaxis.set_major_locator(MultipleLocator(args.sliding_windows))
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=3)

        plt.savefig(os.path.join(args.output_dir, f"{vid}.png"), bbox_inches='tight', dpi=500)
        plt.close()

    output_arr = np.zeros((32, (len(actions_dict) + 1) * 100), dtype=np.uint8)
    for i in range(len(actions_dict)):
        output_arr[:, i * 100: (i + 1) * 100] = i
    output_arr[:, len(actions_dict) * 100:] = 255

    output_img = Image.fromarray(output_arr, mode='L')
    output_img.putpalette(palette)
    plt.figure()
    plt.title('Palette Index')
    plt.gca().xaxis.set_major_locator(MultipleLocator(100))
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=3)
    plt.imshow(output_img)
    plt.savefig(os.path.join(args.output_dir, "palette.png"), bbox_inches='tight', dpi=500)
    plt.close()


if __name__ == "__main__":
    main()
