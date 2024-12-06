![alt text](assets/Banner.png)

# VisionRD AI Hackathon 2024 🧠💻

Welcome to the **VisionRD AI Hackathon 2024** repository! This repo contains all the scripts, architectures, and utilities you'll need to build powerful AI models during the hackathon.

---

## 🏗️ Repository Structure
```plaintext
VisionRD-AI-Hackathon-2024/
│
├── data/                         # Place datasets here
├── arches/                       # Backbone architectures (ResNet, RegNet, DLA)
├── utils/                        # Utility scripts for data visualization and model visualization
├── requirements.txt              # Python dependencies
└── main.py                       # Entry point script

---

## 📂 Data
1. **Download the dataset** using the link provided [here](https://drive.google.com/drive/folders/1WLRThi__ScZdeQQfCOfNDL5xpvo-YUd4?usp=sharing).
2. Place the data files in the `data/` folder.

### Dataset: GTEA (Georgia Tech Egocentric Activity)
The **Georgia Tech Egocentric Activities (GTEA)** dataset contains seven types of daily activities, such as making a sandwich, tea, or coffee. Each activity is performed by four different people, resulting in a total of 28 videos. 

#### Key Details:
- **Number of Videos:** 28
- **Activity Types:** Seven types of daily tasks.
- **Annotations:** Approximately 20 fine-grained action instances per video (e.g., *take bread*, *pour ketchup*).
- **Duration per Video:** ~1 minute.

For more details, refer to the GTEA dataset [website](https://sites.google.com/view/gtea/). Videos should be downloaded separately as instructed.


![alt text](assets/dataset.jpg)
---

## 📜 Backbones and Architectures
- **Arches** contains subfolders for:
  - **ResNet 18, 34, 20, 101**
  - **RegNet 40, 80, 160, 320**
  - **DLA        dla34,dla46,dla60, dla102,
**
  
Each folder includes:
- Utility scripts to initialize and use the backbone.
- Documentation (`README.md`) on configurations and usage.

Refer to `arches/README.md` for an overview.

---

## 🔧 Utility Scripts
- `data_processing.py`: Functions for preprocessing the dataset.
- `model_utils.py`: Tools to load, save, or evaluate models.

Detailed usage instructions are in `utils/README.md`.

---

## 🚀 Quick Start
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/VisionRD-AI-Hackathon-2024.git
   cd VisionRD-AI-Hackathon-2024
