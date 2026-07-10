# Eye Gender Prediction Model

## About
Deep learning CNN model that predicts gender from eye images.

## Dataset
- 9,220 eye images
- 4,162 female, 5,058 male
- 80% training, 20% testing

## Model
- Convolutional Neural Network (CNN)
- 4 convolutional layers
- TensorFlow/Keras
- Test Accuracy: ~85%

## Files
- `eye_gender_model.py` - Training script

## How to Use

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load model
model = load_model('eye_gender_model.h5')

# Load image
img = cv2.imread('eye_image.jpg')
img = cv2.resize(img, (30, 30))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
gender = "Male" if prediction[0][0] >= 0.5 else "Female"
print(f"Predicted: {gender}")
```

## Technologies
- Python 3.11.15
- TensorFlow
- OpenCV
- NumPy

#DeepLearning #MachineLearning #CNN #Python
