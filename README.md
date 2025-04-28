# CCDEPLRL_PROJECT

 This repository is for the project in Deep Learning Course

# Violence Detection System

This system detects violent content in videos using deep learning. The system consists of several components:

1. Frame extraction from videos
2. Model training on extracted frames
3. Violence prediction on new videos

## Directory Structure

The system uses the following directory structure:

```
CCDEPLRL_PROJECT/
├── dataset/
│   ├── Violence/       (Violent videos for training)
│   ├── NonViolence/    (Non-violent videos for training)
│   └── frames/         (Extracted frames, created automatically)
├── input/              (Videos to analyze)
├── output/             (Processed videos with violence detection)
└── model/              (Trained model files)
```

## Requirements

- Python 3.6 or higher
- TensorFlow 2.3.0
- Keras 2.4.3
- OpenCV
- scikit-learn
- numpy
- matplotlib
- tqdm
- imutils

You can install the required packages using:

```
pip install tensorflow==2.3.0 keras==2.4.3 opencv-python scikit-learn numpy matplotlib tqdm imutils
```

## Usage

### Setting up the directory structure

```
python violence_detection_main.py --setup
```

### Adding training data

Place your training videos in the appropriate folders:
- Violent videos in `CCDEPLRL_PROJECT/dataset/Violence/`
- Non-violent videos in `CCDEPLRL_PROJECT/dataset/NonViolence/`

### Extracting frames from videos

```
python violence_detection_main.py --extract
```

### Training the model

```
python violence_detection_main.py --train
```

### Running prediction on videos

Place the videos you want to analyze in the `CCDEPLRL_PROJECT/input/` folder, then run:

```
python violence_detection_main.py --predict
```

### Running prediction on a specific video

```
python violence_detection_main.py --predict --video path/to/your/video.mp4
```

## Output

- Processed videos with violence detection labels will be saved in the `CCDEPLRL_PROJECT/output/` folder
- A summary report with violence percentages for each video will be saved as `CCDEPLRL_PROJECT/violence_detection_report.txt`

## Notes

- Training may take a significant amount of time depending on your hardware and the size of your dataset
- Using a GPU can significantly speed up the training process
- The system works best with videos that have less than 30 seconds duration
