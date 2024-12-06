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

`curl --header 'Host: drive.usercontent.google.com' --header 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' --header 'Accept-Language: en-US,en;q=0.9' --header 'Cookie: SID=g.a000qgiaJtS64IvatQLVaHb9aXIUauyQNNDxvXDXe_BpoHp2uNArJNIeNZ05lbTpdvEkiN9suAACgYKAYASARESFQHGX2MiK65TcSnF9kCwewYWAq9XURoVAUF8yKoSObpJG6teiLNjex563RfH0076; __Secure-1PSID=g.a000qgiaJtS64IvatQLVaHb9aXIUauyQNNDxvXDXe_BpoHp2uNAr2V7fZO5n2jRqOHdSvMYoxwACgYKASkSARESFQHGX2MiZjjdRLVmnRK0anjcDtzlpBoVAUF8yKrOqWQQqZECCwJvgqPegGYM0076; __Secure-3PSID=g.a000qgiaJtS64IvatQLVaHb9aXIUauyQNNDxvXDXe_BpoHp2uNAr9rMVcttTkgVskO0bHxu_agACgYKASgSARESFQHGX2MinGzL-wM4DdAgFTJStnF7zRoVAUF8yKqw5XxIxXo4YapNK2lJ2a4G0076; HSID=AVLiD19r4MCfWyw_Y; SSID=AWyb8D5R4oRi_VGNU; APISID=lapLqAnHgkPxvzZp/AYY2M6d8VMjpWpR99; SAPISID=D51sXsFk3645942C/Azc4zflCO7pVjqNK_; __Secure-1PAPISID=D51sXsFk3645942C/Azc4zflCO7pVjqNK_; __Secure-3PAPISID=D51sXsFk3645942C/Azc4zflCO7pVjqNK_; AEC=AZ6Zc-WyT7DH3Zh1c4jQErrQzQKs9L_o2G3RfdQ6ju1qJW5x9D7SQk8w244; NID=519=XHglq5iaC1v9zybmKuwhvovrQtPsS0_4_fk4Nq8z0xPFPru8pBlWqZmZqNfiiyGsjkBhMnprnuLmSaFVSzn55FnvqRZaTnZDFewdEuSq1cvLU2D5pHLBTWLyVW0As5LtpqsMoVitemZ5xDTs1FWvm_vtotrtNt9UsVlqjVHD_atcK5ZAH8LPes61u9X2aXS5QTs-PK6AScZTUcUQCR0TdBXkJjvUX7YfydmWk6TU88pDFK2XE_MxEPUxF2qA3yURTaMKP7FaQqsHfZaMTohcKJiyEtaeGALSRKWYxyvQhjQmZA3T2dGfFgLYWKhLzgFkTRuyr3FjBTOfUajqAiPkRk_O3wKv99kO0792MJ48fxNyVlO9tig5ZJ3bg-dCE_v3dTJho85scODRkUa_AuaPVm0Fj-RsbuGO_B__s_4Q8DJS1TFSmNk3k4yv8IwBRWe7Fp4gVPhbQ_ooLM9fe7uIRb_7wSDek6fNyptQ6cdrSuFCvPt1h03hPW_-rfmQVBYQ5bPc5Etg8NHMFFW5ayM4CAcyn00YQNpN3ZqS3llpK1JZp2v2veEpJy_OnD3nsgJnWmxpk1ZQdy0sUgklhg3fMeCecq5sNuuCZhg0PAlm_f93fmdszaw0vy3FIw1T7bWmhykpVGHF1z4OYZgXpguJWDqq_EDVElpMTqrhKN_e5QfDq133oWQoiYfJL6DXmhCPWw9M9BHxKSsuN-0MfciJvmFGvHUspVv1dcdYzF6HtLySPBpvlaXru2W8scJF4mPMctKEVrempV7Ve8G2emCi6zr9DfFPfcDm8gRTmS-WEWAkPLrDQp6e4eIe3sHjQv0sNe2F3PuGh-ZXxhQ9G-_SL3c6WUrLrqlmHlorF3Q3cw; __Secure-1PSIDTS=sidts-CjEBQT4rX1ykuJRBvNkRU4F0dHmZawI4Baziwl_SfGQ3bQi2rbYcaqC9Lwg1NFIh6uMREAA; __Secure-3PSIDTS=sidts-CjEBQT4rX1ykuJRBvNkRU4F0dHmZawI4Baziwl_SfGQ3bQi2rbYcaqC9Lwg1NFIh6uMREAA; SIDCC=AKEyXzX2I4wI19nay3rtztBpxwyxQ_FsDlafYC5KlA2JG5NbrdzSwr1jtdR2AEyEkT2028a1V5gu; __Secure-1PSIDCC=AKEyXzWz3yWU0dEbKWVpXuOs7KzeJp3m5aha5hcIEA6c22b8L07dtoAcGd2l51IDQeAHRtq9cxRR; __Secure-3PSIDCC=AKEyXzWjHMQ8L9UkqN1XXiZM5k98uuxXfO6Qt7pRsueAmFHs8azX5QWsDqDBFigxgTr_M1yxuuCq' --header 'Connection: keep-alive' 'https://drive.usercontent.google.com/download?id=1S8IZ1Q4TRbo84M49BcX7Klip9eMP5QlM&export=download&authuser=0&confirm=t&uuid=16f76d2b-7cd0-4011-b889-4fb2930bc4fa&at=APvzH3qW9qOOKoxiHiQTV_CaVK9f%3A1733493999797' -L -o 'gtea.zip'`

3. Place the data files in the `data/` folder.

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


#### Folder Structure
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

