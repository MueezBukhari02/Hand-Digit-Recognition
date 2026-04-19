# Deep Learning Projects

A collection of deep learning projects built from scratch using TensorFlow/Keras, Scikit-Learn, OpenCV, and Python.

**Author:** Syed Mueez Ul Hassan Bukhari

---

## Projects

### 1. Hand Digit Recognizer
> Draw a digit using your mouse and a CNN model predicts what digit it is in real time.

- Trained a Convolutional Neural Network on the MNIST dataset (60,000 images)
- Achieved **99.2% validation accuracy**
- Built an interactive desktop app using OpenCV where users draw digits with their mouse
- Model exported from Google Colab and deployed locally
- Includes confidence scoring and "not a digit" detection for invalid inputs

**Tech stack:** TensorFlow/Keras, OpenCV, NumPy, Python

📁 [View Project](./hand-digit-recognition/)

---

## Skills Demonstrated

- Building and training CNNs from scratch
- Data preprocessing and normalization
- Model evaluation and visualization (accuracy/loss curves)
- Exporting and loading trained models
- Building real-time interactive computer vision apps with OpenCV

---

## Setup

Each project has its own folder with a dedicated `requirements.txt`. To run any project:

```bash
cd project-folder
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python app.py
```

---

## Environment

- Python 3.11
- TensorFlow 2.x
- Google Colab (training)
