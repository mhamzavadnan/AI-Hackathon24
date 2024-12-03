# Backbone Models

This directory contains implementations of various backbone architectures for deep learning tasks. Each backbone model is implemented as a standalone Python module and can be used interchangeably for different tasks.

---

## **Available Backbones**6

1. **PResNet** (`presnet.py`): A customizable residual network architecture supporting `ResNet18`, `ResNet50`, and `ResNet101`.
2. **DLA34** (`dla34.py`): Deep Layer Aggregation (DLA) architecture for efficient feature reuse.
3. **RegNet** (`regnet.py`): A modern backbone for efficient neural network scaling.

---

## **Input Requirements**

All backbone models take a **4D Tensor** as input:
- **Shape**: `(batch_size, channels, height, width)`
- **Expected Input**: RGB images normalized to a range of [0, 1] or [-1, 1], depending on the pretrained model.
- **Input Channel**: 3 (default for RGB images).

---

## **Output Details**

### **1. PResNet (`presnet.py`)**
- Supports `ResNet18`, `ResNet50`, and `ResNet101` by setting the `depth` parameter to `18`, `50`, or `101`.
- **Return Type**: List of feature maps from selected layers (controlled by `return_idx`).
- **Output Shapes**:
  - Each element in the output list corresponds to a specific layer's output.
  - Feature map resolutions are progressively reduced based on the ResNet stride (4, 8, 16, 32).
  - Example for input shape `(batch_size, 3, 224, 224)`:
    - `ResNet18`: `[torch.Size([batch_size, 64, 56, 56]), ..., torch.Size([batch_size, 512, 7, 7])]`
    - `ResNet50`: Similar to `ResNet18` but with larger feature dimensions due to `BottleNeck`.
    - `ResNet101`: Similar to `ResNet50` with deeper layers.

### **2. DLA34 (`dla34.py`)**
- **Return Type**: List of feature maps from selected levels (controlled by `return_index`).
- **Output Shapes**:
  - Each element in the output list corresponds to a DLA level.
  - Example for input shape `(batch_size, 3, 224, 224)`:
    - `[torch.Size([batch_size, 128, 56, 56]), ..., torch.Size([batch_size, 512, 7, 7])]`

### **3. RegNet (`regnet.py`)**
- **Return Type**: Selected hidden states from the RegNet model (controlled by `return_idx`).
- **Output Shapes**:
  - Each element corresponds to a hidden state from the backbone layers.
  - Example for input shape `(batch_size, 3, 224, 224)`:
    - `[torch.Size([batch_size, 128, 56, 56]), ..., torch.Size([batch_size, 1024, 7, 7])]`

---

## **Usage**

### **1. PResNet**
```python
import torch
from backbone import *

model = PResNet(depth=101, pretrained=True, return_idx=[0, 1, 2, 3])

input_tensor = torch.randn(8, 3, 224, 224)

output = model(input_tensor)

for idx, feature_map in enumerate(output):
    print(f"Feature Map {idx}: {feature_map.shape}")
```

### **2. DLA34**
```python
import torch
from backbone.dla34 import DLANet

model = DLANet(return_index=[2, 3, 4])

input_tensor = torch.randn(8, 3, 224, 224)
output = model(input_tensor)

for idx, feature_map in enumerate(output):
    print(f"Level {idx + 2} Feature Map: {feature_map.shape}")
```

### **3. RegNet**
```python
import torch
from backbone.regnet import RegNet

configuration = {
    "num_classes": 1000,  # Number of output classes
    "input_channels": 3   # Input channels (RGB images)
}

model = RegNet(configuration=configuration, return_idx=[0, 1, 2])

input_tensor = torch.randn(8, 3, 224, 224)
output = model(input_tensor)

for idx, hidden_state in enumerate(output):
    print(f"Hidden State {idx}: {hidden_state.shape}")
```

## **Run Testing Scripts**

- cd C:\VisionRD\AI-Hackathon24\arch> 
- python -m backbone.resnet_testing

---

## **Customization**

### **Pretrained Models**
- Models can be loaded with pretrained weights by setting `pretrained=True`.
- For RegNet, weights are loaded from the HuggingFace repository.

### **Selecting Features**
- The indices of the output layers can be customized using:
  - `return_idx` for PResNet and RegNet.
  - `return_index` for DLA34.

---

## **Dependencies**
- Python >= 3.8
- PyTorch >= 1.7
- Transformers (for RegNet)

---


## **FAQ**

### Q1. How do I use these backbones in my own task?
- Import the desired backbone module.
- Pass an input tensor of shape `(batch_size, 3, height, width)`.
- Extract the output feature maps for downstream tasks like classification, segmentation, or detection.

### Q2. What are the advantages of each backbone?
- **PResNet**: Flexible and supports multiple ResNet depths (18, 50, 101).
- **DLA34**: Efficient feature reuse with hierarchical structure.
- **RegNet**: Scalable and efficient for modern architectures.