# Hand Digit Recognizer

A real-time digit recognition application powered by a Convolutional Neural Network (CNN) trained on the MNIST dataset. Draw any digit (0–9) on the canvas using your mouse and the model instantly predicts what digit it is — along with a confidence score.

![Python](https://img.shields.io/badge/Python-3.11-yellow)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit-learn.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-maroon)
![Accuracy](https://img.shields.io/badge/Accuracy-99.2%25-brightgreen)

---

## Demo

| Draw a digit | Get instant prediction |
|---|---|
| Use your mouse to draw any digit on the black canvas | Press SPACE and the CNN predicts the digit with confidence % |

---

## Features

- **99.2% validation accuracy** on the MNIST test set
- Real-time interactive drawing canvas built with OpenCV
- Confidence score displayed with every prediction
- Smart "Not a digit" detection using entropy and stroke analysis
- Clean and minimal UI — no extra dependencies or frameworks needed
- Trained on Google Colab, deployed locally

---

## How It Works

```
User draws digit
      ↓
Canvas captured (500×500 px)
      ↓
Preprocessed → Grayscale → Crop → Resize to 28×28 → Normalize
      ↓
Fed into trained CNN model
      ↓
Softmax output → 10 probabilities (one per digit)
      ↓
Predicted digit + confidence displayed on screen
```

---

## Training Results

| Metric | Value |
|---|---|
| Training Accuracy | 99.1% |
| Validation Accuracy | 99.2% |
| Test Loss | ~0.03 |
| Epochs | 10 |
| Batch Size | 64 |
| Dataset | MNIST (60,000 train / 10,000 test) |

Training was done on Google Colab with GPU acceleration. The model converged cleanly with no overfitting — training and validation accuracy stayed within 0.1% of each other throughout.

---

## Project Structure

```
hand-digit-recognition/
│
├── model/
│   └── digit_model.keras     # Trained CNN model
│
├── app.py                    # Main application (drawing + inference)
├── requirements.txt          # Dependencies
└── README.md
```

---

## Setup & Run

### Installation

```bash
# Clone the repo
git clone https://github.com/MueezBukhari02/deep-learning-projects.git
cd deep-learning-projects/hand-digit-recognition

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
python app.py
```

### Controls

| Key | Action |
|---|---|
| Mouse drag | Draw digit on canvas |
| `SPACE` | Predict the drawn digit |
| `C` | Clear canvas |
| `Q` or `✕` | Quit |

---

## Dependencies

```
tensorflow-cpu
opencv-python
numpy
```

---

## Limitations

- Model is trained on MNIST which contains relatively clean, centered digits — highly stylized handwriting may reduce accuracy


---

## Author

**Syed Mueez Ul Hassan Bukhari**
GitHub: [@MueezBukhari02](https://github.com/MueezBukhari02)
