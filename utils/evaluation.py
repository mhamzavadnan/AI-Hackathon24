import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.data import CocoEvaluator
from src.misc import MetricLogger, SmoothedValue, reduce_dict
from sklearn.metrics import accuracy_score

# Common Functions for MSTCN and ASFormer
def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
    return content

def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends

def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], np.float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)

    if norm:
        return D[m_row, n_col] / max(m_row, n_col)
    else:
        return D[m_row, n_col]

# Action Loss Function (classification)
def action_loss_fn(predictions, targets):
    """
    Compute the loss for action classification.
    Assumes predictions are logits and targets are integer labels.
    """
    criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss for classification
    return criterion(predictions, targets)

# Combined train function for action, bounding box, and keypoints
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    # Tracking total action accuracy and losses
    total_action_loss = 0
    total_action_accuracy = 0
    total_bbox_loss = 0
    total_keypoint_loss = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

        metric_logger.update(loss=loss.item(), **loss_dict)

        # Assuming 'outputs' contains action predictions, bounding box, and keypoints:
        # Action Loss and Accuracy
        action_preds = outputs['action_logits']
        action_targets = targets['action']
        
        action_loss = action_loss_fn(action_preds, action_targets)
        total_action_loss += action_loss.item()
        
        # Calculate action accuracy
        _, action_pred_classes = action_preds.max(1)
        action_accuracy = (action_pred_classes == action_targets).float().mean().item()
        total_action_accuracy += action_accuracy

        # Bounding Box and Keypoint Loss (example placeholders, replace with actual logic)
        bbox_loss = loss_dict.get('bbox_loss', 0)
        keypoint_loss = loss_dict.get('keypoint_loss', 0)
        total_bbox_loss += bbox_loss.item()
        total_keypoint_loss += keypoint_loss.item()

    # Logging the results
    total_action_accuracy /= len(data_loader)
    total_action_loss /= len(data_loader)
    total_bbox_loss /= len(data_loader)
    total_keypoint_loss /= len(data_loader)

    print(f"Epoch {epoch}: Total Action Loss: {total_action_loss:.4f}, "
          f"Action Accuracy: {total_action_accuracy:.4f}, "
          f"Bounding Box Loss: {total_bbox_loss:.4f}, "
          f"Keypoint Loss: {total_keypoint_loss:.4f}")

    # Optionally update EMA
    if ema is not None:
        ema.update(model)

# Argument parser for training configuration
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="Directory containing input data")
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="Directory to save output")
    parser.add_argument('--max_norm', type=float, default=0, help="Gradient clipping max norm")
    return parser.parse_args()

def main():
    args = get_args()
    # Initialize your model, criterion, optimizer, etc. here
    # model = ...
    # optimizer = ...
    # data_loader = ...
    # criterion = ...
    
    for epoch in range(1, 101):  # Example for 100 epochs
        train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm=args.max_norm)

if _name_ == "_main_":
    main()