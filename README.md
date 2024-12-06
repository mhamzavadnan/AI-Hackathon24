![alt text](assets/Banner.png)

# VisionRD AI Hackathon 2024 🧠💻

Welcome to the **VisionRD AI Hackathon 2024** repository! This repo contains all the scripts, architectures, and utilities you'll need to build powerful AI models during the hackathon.

---

## 🏗️ Repository Structure
```plaintext

VisionRD-AI-Hackathon-2024/
│
├── data/                         # Place datasets here
├── arches/backbone               # Backbone architectures (ResNet, RegNet, DLA)
├── utils/                        # Utility scripts for data visualization and model visualization
├── requirements.txt              # Python dependencies
└── main.py                       # Entry point script
└── requirements.txt                

```
---

<div style="display: flex; align-items: center;">

<div style="flex: 1; padding-right: 10px;">
  
### Dataset: GTEA (Georgia Tech Egocentric Activity)

The **Georgia Tech Egocentric Activities (GTEA)** dataset contains seven types of daily activities, such as making a sandwich, tea, or coffee. Each activity is performed by four different people, resulting in a total of 28 videos. 

#### Key Details:
- **Number of Videos:** 28
- **Activity Types:** Seven types of daily tasks.
- **Annotations:** Approximately 20 fine-grained action instances per video (e.g., *take bread*, *pour ketchup*).
- **Duration per Video:** ~1 minute.

For more details, refer to the GTEA dataset [website](https://sites.google.com/view/gtea/). Videos should be downloaded separately as instructed.

1. **Download the dataset** using the link provided [here](https://drive.google.com/drive/folders/1WLRThi__ScZdeQQfCOfNDL5xpvo-YUd4?usp=sharing).
2. Place the data files in the `data/` folder.

</div>

<div style="flex: 1;">
  <img src="assets/dataset.jpg" alt="Dataset Image" style="max-width: 100%; height: auto;">
</div>

</div>



---

## 📜 Backbones and Architectures

The `arches/` directory contains subfolders for various backbone architectures. These backbones are modular and can be utilized for a wide range of AI applications, including action recognition.

### Available Architectures
#### ResNet
- **Variants:** ResNet-18, ResNet-34, ResNet-50, ResNet-101
- Description: ResNet (Residual Networks) introduces skip connections to solve the vanishing gradient problem, enabling the training of deeper networks. 

#### RegNet
- **Variants:** RegNet-40, RegNet-80, RegNet-160, RegNet-320
- Description: RegNet (Regular Networks) optimizes network design by varying the depth and width of layers, resulting in efficient architectures.

#### DLA (Deep Layer Aggregation)
- **Variants:** DLA-34, DLA-46, DLA-60, DLA-102
- Description: DLA combines information across layers in a hierarchical manner, improving feature aggregation and task performance.

---

### Folder Structure
Each architecture folder contains the following:
- Model definitions and initialization scripts.
- Pretrained weights (if applicable).
- Usage examples documentation in a `README.md` file.
```plaintext
Example:
arches/backbone
│
├── README.md
├── resnet.py
├── regnet.py
├── dla.py
```
Refer to `arches/backbone/README.md` for an overview.

---

## 🔧 Utility Scripts

This folder contains scripts for various utility functions like data processing, model evaluation, and visualization. Below is an overview of the available scripts:

### File Structure
```plaintext
utils/
├── README.md                               # Documentation for utility scripts
├── action_visualization.py                 # Visualize action predictions
├── arch_visualization.py                   # Visualize model architectures
├── data_visualization.py                   # Visualize dataset samples
├── feature_manifold_visualization.py       # Visualize feature manifolds
├── generate_all_files.py                   # Generate required files for pipelines
├── labels_visualization.py                 # Visualize labels and annotations
├── prediction_visualization.py             # Visualize model predictions
├── splitter.py                             # Split datasets into training, validation, and testing
├── xml_to_coco.py                          # Convert XML annotations to COCO format
├── data_processing.py                      # Functions for preprocessing the dataset
└── model_utils.py                          # Tools to load, save, or evaluate models
```

Detailed usage instructions are in `utils/README.md`.

---

## 🚀 Quick Start

Follow these steps to get started with the VisionRD AI Hackathon repository:

### 1. Clone the Repository
First, clone this repository to your local machine:
```bash
git clone https://github.com/<your-username>/VisionRD-AI-Hackathon-2024.git
cd VisionRD-AI-Hackathon-2024
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset
```plaintext
VisionRD-AI-Hackathon-2024/
└── data/
    └── GTEA/
```


This markdown provides clear instructions for setting up the repository and starting the project, ensuring participants have a seamless onboarding experience. Let me know if you'd like further adjustments! Start building your custom architectures, writing training & inference scripts, and logging your evaluations on tensorboard!!

