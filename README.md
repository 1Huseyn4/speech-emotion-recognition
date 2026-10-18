# Speech Emotion Recognition

Recognizes human emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) from speech audio using speech signal processing (MFCC features) and machine learning / deep learning models.

## Project Overview

- **Dataset:** RAVDESS speech recordings (1440 files, 24 actors, 8 emotions)
- **Features:** 13 Mel-Frequency Cepstral Coefficients (MFCCs) extracted with Librosa
- **Models:**
  - **SVM (baseline):** RBF-kernel SVM on the mean and standard deviation of each MFCC
  - **CNN:** 2D convolutional network on the full 13 x 200 MFCC matrix, with batch normalization, time/frequency masking augmentation and early stopping
- **Evaluation:** speaker-independent (actors in the test set are never seen during training)

## Technologies Used

Python, Librosa, NumPy, Pandas, Scikit-learn, PyTorch, Matplotlib

## Dataset

This project uses the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), speech part (`Audio_Speech_Actors_01-24.zip`).

Download it from the official source: <https://zenodo.org/records/1188976>

Extract it so the files look like `data/Actor_01/03-01-01-01-01-01-01.wav`. The `data/` folder is not included in this repository.

## Project Structure

```
speech-emotion-recognition/
├── src/
│   ├── utils.py                # constants, seeding, actor-independent split, MFCC helpers
│   ├── model.py                # CNN architecture (shared by training and prediction)
│   ├── make_metadata.py        # scan the dataset and build data/processed/metadata.csv
│   ├── build_features.py       # MFCC mean/std features for the SVM
│   ├── build_cnn_features.py   # padded MFCC matrices for the CNN
│   ├── train_model.py          # train and evaluate the SVM
│   ├── train_cnn.py            # train and evaluate the CNN
│   ├── predict_emotion.py      # predict the emotion of any WAV file
│   └── explore_audio.py        # waveform and MFCC exploration of one file
├── results/                    # confusion matrices, reports, metrics
├── requirements.txt
└── README.md
```

## How to Run

Run all commands from the repository root.

1. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Build the metadata table:

   ```
   python src/make_metadata.py
   ```

3. Extract the features (both are needed: one for the SVM, one for the CNN):

   ```
   python src/build_features.py
   python src/build_cnn_features.py
   ```

4. Train and evaluate the models:

   ```
   python src/train_model.py
   python src/train_cnn.py
   ```

5. Predict the emotion of an audio file:

   ```
   python src/predict_emotion.py data/Actor_01/03-01-01-01-01-01-01.wav
   ```

Training scripts write confusion matrices, classification reports and `*_metrics.json` files to `results/`, and the trained models to `models/`.

## Evaluation Protocol

- The dataset is split **by actor**: 20% of the actors form the test set and no actor appears in both training and test data.
- The SVM and the CNN use **exactly the same test actors** (same seed, same split function), so their results are directly comparable.
- The CNN additionally holds out a few training actors as a validation set, used only for early stopping. MFCC normalization statistics are computed on training data only.
- Because the classes are slightly imbalanced (`neutral` has half as many samples as the others), both accuracy and macro-F1 are reported.

## Results

Test set: held-out actors, speaker-independent. Chance level for 8 classes is 12.5%.

| Model | Accuracy | Macro-F1 |
| ----- | -------- | -------- |
| SVM (MFCC mean/std) | XX.XX% | XX.XX% |
| CNN (MFCC matrix)   | XX.XX% | XX.XX% |

Detailed outputs are in `results/`:

- `svm_confusion_matrix.png`, `svm_classification_report.txt`, `svm_metrics.json`
- `cnn_confusion_matrix.png`, `cnn_classification_report.txt`, `cnn_metrics.json`, `cnn_training_curve.png`

## Limitations and Future Improvements

- RAVDESS contains only 24 actors, so speaker-independent accuracy is limited and varies with the choice of test actors. Cross-validation over actor groups (`GroupKFold`) would give a more stable estimate.
- Only MFCCs are used. Delta features, chroma or mel-spectrograms may help.
- A recurrent model (LSTM / GRU) or a pretrained audio model (e.g. wav2vec 2.0) could be compared against the CNN.
- Evaluation on additional datasets (TESS, EMO-DB) would test generalization across recording conditions.
