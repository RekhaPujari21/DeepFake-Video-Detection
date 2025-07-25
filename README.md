# 🤖 DeepFake Video Detection using TensorFlow & InceptionV3

This project is focused on detecting **DeepFake videos** using a Deep Learning pipeline that combines **InceptionV3** for feature extraction and **GRU (Gated Recurrent Units)** for temporal sequence modeling. It identifies whether a given video is **REAL** or **FAKE** based on facial movements and inconsistencies.

---

## 📁 Dataset Structure

The dataset should be organized as follows:

dataset/
├── train_sample_videos/
│ ├── [video1].mp4
│ ├── [video2].mp4
│ └── metadata.json
└── test_videos/
├── [test_video1].mp4
└── ...


- `metadata.json`: Includes the label (`REAL` or `FAKE`) for each training video.
- The test videos are unlabeled and used for inference.

-------------------------------------------------------------------------------------------
## 🧰 Requirements

Install required packages using pip:


- pip install tensorflow
- pip install opencv-contrib-python
- pip install imageio
- pip install matplotlib
- pip install pandas
------------------------------------------------------------------------------------------
## ⚙️ How It Works
1. 📹 Video Frame Extraction
- Extracts 20 frames from each video using OpenCV.

- Frames are center-cropped and resized to 224x224 pixels.

- RGB normalized for consistency.

2. 🔍 Feature Extraction
- Uses a pretrained InceptionV3 model (without top layer).

- Converts each frame into a 2048-dimensional feature vector.

3. ⏩ Sequence Modeling
- Sequence of features passed into:

- GRU(16) → GRU(8)

- Followed by a Dropout and Dense layers

- Final layer is a sigmoid activation that outputs:

  - FAKE if score ≥ 0.5

  - REAL if score < 0.5

4. Training
- Binary classification using binary_crossentropy.

- Optimizer: Adam

- Uses ModelCheckpoint to save the best model based on accuracy.
---------------------------------------------------------------------------------
## 📊 Visualizations
✅ Label Distribution (REAL vs FAKE) using Matplotlib bar chart.

🖼️ Frame preview and processing shown using matplotlib.pyplot.

🎞️ Inline video display using IPython HTML (for notebooks).
--------------------------------------------------------------------------------
## 🚀 Inference Example
- After training the model, you can predict any test video like this:
''' bash 
if sequence_prediction('bwbp1.mp4') >= 0.5:
    print("Predicted: FAKE")
else:
    print("Predicted: REAL")
----------------------------------------------------------------------------------




