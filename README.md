# Facial-Expression-Recognition

## Overview
This project implements a facial expression recognition system using a Convolutional Neural Network (CNN). The system can detect and classify facial expressions in real time from a live video feed.

## Features
- **Model Training**: Train a CNN on facial expression datasets.
- **Real-Time Recognition**: Use a webcam to recognize emotions in real time.
- **Supported Emotions**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

## Prerequisites
Before running the project, ensure the following are installed:

- Python 3.8 or above
- TensorFlow
- OpenCV
- NumPy
- scikit-learn

Install dependencies using:
```bash
pip install tensorflow opencv-python numpy scikit-learn
```

## Dataset
- The training data should be organized into folders, with each folder representing a class (e.g., `Happy`, `Sad`).
- Images should be grayscale and resized to 48x48 pixels for model training.
- Dataset link 
```link
https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset
```

## How to Use

### Step 1: Model Training
1. Place your dataset in a structured format (e.g., `dataset_folder/Happy`, `dataset_folder/Sad`).
2. Update the `image_dir` variable in the training script to the path of your dataset.
3. Run the training script:
   ```bash
   python train_model.py
   ```
4. The trained model will be saved as `facial_expression_model.h5`.

### Step 2: Real-Time Recognition
1. Ensure the trained model (`facial_expression_model.h5`) is in the same directory as the recognition script.
2. Run the recognition script:
   ```bash
   python real_time_recognition.py
   ```
3. A webcam window will open, showing live video with detected emotions.
4. Press **'q'** to exit the video feed.

## Project Structure
```
project_folder/
|-- train_model.py            # Script for training the CNN
|-- real_time_recognition.py  # Script for real-time emotion detection
|-- facial_expression_model.h5 # Trained model file
|-- README.md                 # Project documentation
|-- train/           # Folder containing training images
```

## Customization
- Add new classes by including labeled folders in your dataset and retraining the model.
- Adjust hyperparameters (e.g., epochs, batch size) in the training script for optimization.

## Notes
- Ensure good lighting for the webcam during real-time recognition.
- The Haar Cascade XML file (`haarcascade_frontalface_default.xml`) is included in OpenCV.

## References
- [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- OpenCV Documentation
- TensorFlow/Keras Documentation

## License
This project is licensed under the MIT License. Feel free to use and modify it as needed.

---
For any issues or questions, please contact [Your Name] at [Your Email].

