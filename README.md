# 🗣️ Real-Time Sign Language Translator with Voice

A computer vision project that translates hand sign language into text and speech in real-time. Built with Python, OpenCV, and Scikit-Learn.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 🚀 Features
- **Real-Time Detection:** Uses MediaPipe to track 21 hand landmarks at 30+ FPS.
- **Machine Learning Classifier:** Compares Random Forest vs. Neural Networks for optimal latency (Random Forest chosen for 40% faster inference).
- **Smart Voice Output:** Uses `pyttsx3` with a stability filter to speak the predicted sign without "stuttering."
- **Relative Coordinates:** Custom data preprocessing ensures the model works regardless of hand position or distance from the camera.

## 🛠️ Tech Stack
- **OpenCV:** Video capture and frame processing.
- **MediaPipe:** Hand landmark extraction.
- **Scikit-Learn:** Model training (Random Forest Classifier).
- **Pyttsx3:** Text-to-speech engine.

## ⚙️ How to Run
1. Clone the repo:
    git clone [https://github.com/rakshitjainn/Sign-Language-Translator-Pro.git](https://github.com/rakshitjainn/Sign-Language-Translator-Pro.git)
2. Install dependencies:
    pip install -r requirements.txt
3. Run the App:
    python main.py

## 🧠 Engineering Decisions
    Why Random Forest? I benchmarked a Multi-Layer Perceptron (Neural Net) against a Random Forest. While both achieved ~99% accuracy on the test set, the Random Forest had significantly lower inference time on CPU, making it better suited for real-time video processing on standard laptops.

    Addressing Drift: To prevent the voice engine from spamming words, I implemented a "Stability Buffer" that requires a sign to be held for 10 consecutive frames (approx 0.3s) before triggering the voice output.

Built by Rakshit Jain