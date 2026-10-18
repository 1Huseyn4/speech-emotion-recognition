# Speech Emotion Recognition

## Project Overview
This project recognizes human emotions from speech audio using machine learning and deep learning techniques. It uses the RAVDESS dataset and extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio signals.

## Technologies Used
- Python
- Librosa
- NumPy and Pandas
- Scikit-learn
- PyTorch
- Matplotlib

## Models
- **Support Vector Machine (SVM):** Uses statistical MFCC features.
- **Convolutional Neural Network (CNN):** Learns patterns from MFCC features.

## Dataset
The project uses the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).

Download the dataset from the official source: https://zenodo.org/records/1188976

The dataset files are stored locally and are not included in this repository.

## Project Structure
```text
speech-emotion-recognition-1/
├── src/
│   ├── make_metadata.py
│   ├── build_features.py
│   ├── build_cnn_features.py
│   ├── extract_features.py
│   ├── train_model.py
│   ├── train_cnn.py
│   └── predict_emotion.py
├── results/
│   └── svm_confusion_matrix.png
├── requirements.txt
└── README.md
```

## How to Run
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare the dataset metadata:
   ```bash
   python src/make_metadata.py
   ```
3. Extract MFCC features:
   ```bash
   python src/build_features.py
   ```
4. Train the SVM model:
   ```bash
   python src/train_model.py
   ```
5. Train the CNN model:
   ```bash
   python src/train_cnn.py
   ```
6. Predict an emotion from an audio file:
   ```bash
   python src/predict_emotion.py
   ```

## Results
The CNN achieved approximately 52.78% accuracy in the current experiment. The SVM achieved approximately 36.67% accuracy in a separate experiment. Since the models used different train/test splits, these results should not be treated as a direct comparison.

## Future Improvements
- Use consistent actor-independent train/test splits for fair model comparison.
- Improve generalization to unseen speakers.
- Evaluate models with additional metrics.
