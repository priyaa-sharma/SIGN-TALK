# Sign Language Interpreter using Machine Learning

Real-time static sign recognition from a webcam. Converts hand signs into
text (and speech).

## Pipeline
Webcam -> MediaPipe Hands (21 landmarks) -> normalisation (63 features)
-> Random Forest classifier -> temporal smoothing -> text + speech

## Setup
    pip install -r requirements.txt

## Run
    python collect_data.py --labels A B C D E HELLO THANKS SPACE DEL   # 1. record data
    python train.py                                                    # 2. train
    python app.py                                                      # 3. interpret live

## Why landmarks instead of raw images?
Landmarks ignore background, lighting and skin tone, need far less data,
and run in real time on a normal laptop CPU.

## Limitations / future work
- Static signs only (ASL J and Z need motion) -> add an LSTM over landmark sequences
- Single hand -> extend to two hands and facial expressions
- Own-collected data -> try a public dataset (ASL Alphabet, WLASL, ISL datasets)
- Compare models: SVM, KNN, MLP, CNN; add a web/mobile front end
