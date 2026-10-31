

## SignTalk: Sign Language Interpreter using Machine Learning

SignTalk turns hand signs into text and spoken words in real time, using only a normal webcam. It recognises the full alphabet (A to Z) and five movement-based words, and it speaks every letter and word out loud with a natural AI voice. There is nothing to press while you use it.


## Features
Alphabet A to Z recognised from still hand shapes. Hold a sign for about a second and the letter is added to the sentence.
Words with movement (HELLO, THANKYOU, SORRY, PLEASE, ILOVEYOU), recognised hands-free. Just sign them whenever you like.
Automatic AI voice. Every letter and word is spoken as soon as it is detected, using Microsoft neural voices (edge-tts).
Train it yourself. Record your own signs from the webcam and retrain in seconds.

<<<<<<< HEAD
=======
## WORKING 

https://github.com/user-attachments/assets/a24c0c7b-8b6b-4566-a791-6be3590a682b



>>>>>>> a7202ba7d39be1f2dac75f6baeb6a197b7bf6196
## How it works
Webcam
  -> MediaPipe HandLandmarker (21 hand landmarks)
  -> Letters: normalise 63 features -> Random Forest -> smoothing -> hold 1 s -> letter
  -> Words:   last 2 seconds of landmarks + wrist movement -> Extra Trees -> word
  -> text on screen + AI voice


##Setup

Requires Python 3.10 or newer and a webcam. The voice playback and fonts use Windows features, so the app is written for Windows.

bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

The first run downloads the hand landmark model (about 8 MB) automatically, and the AI voice needs an internet connection.

## Usage

1. Record the letters (about 200 to 300 samples per sign; press S to start and stop, N for the next letter, Q to quit)

bash
python collect_data.py --labels A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
python train.py

2. Record the words (press R for each 2-second clip, about 30 clips per word)

bash
python record_words.py --labels HELLO THANKYOU SORRY PLEASE ILOVEYOU
python record_words.py --labels NONE
python train_words.py

For NONE, do things that are not words, with your hand in view: hold letter signs, wave, move your hand around, rest it, lift it and lower it.

### Optional: add the SignMatic word vocabulary

The Kaggle SignMatic dataset contains 50 isolated ASL words plus an `Idle` class.
It adds word recognition, not recognition of 50 complete phrases. After
extracting it to `data/signmatic/SignMatic_ASL_Keypoint_Dataset_Kaggle`, train
the additional classifier:

```powershell
python train_signmatic.py
```

The training script reports accuracy on a random holdout of real samples; this
is not a signer-independent evaluation. Then start the app as usual with
`python app.py`. The first run downloads a MediaPipe pose landmark model. The
existing alphabet and locally recorded phrase models remain available.

The dataset is licensed under CC BY 4.0. Attribute Joe Elie Sarkis and Ali
Mohamad Yaghi, “SignMatic ASL Keypoint Dataset: 50 Words and Idle” (2026).
See the [Kaggle dataset](https://www.kaggle.com/datasets/joesarkis/signmatic-asl-keypoint-dataset-50-words-and-idle)
for its documentation and license.

3. Run it

bash
python app.py

Limitations

Letters use still shapes only, so the model cannot read signs that depend on movement.
It tracks one hand at a time.
The models learn from the signer who recorded them, so accuracy drops for a different person, lighting or camera distance. Re-recording a few clips fixes this.
The signs are ASL. Indian Sign Language and other sign languages use different signs.
The AI voice needs an internet connection.


## Future work

Replace the word model with an LSTM over landmark sequences
Hindi output and Indian Sign Language (ISL) support
Two-hand signs and facial expressions
Compare more models (SVM, KNN, MLP) and test with more signers
A web or mobile version


## Tech stack: Python, MediaPipe, OpenCV, scikit-learn, Pillow, edge-tts
