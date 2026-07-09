import os
import zipfile
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Flatten, MaxPooling2D, Input,
    Conv2D, GlobalAveragePooling2D )

#Using pandas to import the training data csv file
Csv_file = pd.read_csv(r"C:\Users\USER\Desktop\RDA 212_Exam\eye_gender_data\eye_gender_data\Training_set.csv")
Csv_file.head() 

#importing training dataset and resizing all images 
Train_data_folder = (r"C:\Users\USER\Desktop\RDA 212_Exam\eye_gender_data\eye_gender_data\train")
all_images = os.listdir(Train_data_folder)
print(f"Found {len(all_images)} files in folder")
print(f"  First 5 files: {all_images[:5]}")

Image_size   = (30, 30) 

# Loading images and labels
X, y, loaded_filenames = [], [], []
 
for idx, imgFile in enumerate(all_images):
    imgPath = os.path.join(Train_data_folder, imgFile)
    
    # Skip non-image files
    if not imgFile.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        continue
    
    try:
        # Read image
        image = cv2.imread(imgPath)
        if image is None:
            print(f"Skipping {imgFile}: Could not read image")
            continue
        
        # Resize
        image = cv2.resize(image, Image_size)
        
        # Normalize (0-1)
        image = image / 255.0
        
        # Find label from CSV
        row = Csv_file[Csv_file["filename"] == imgFile]
        
        if row.empty:
            print(f"Skipping {imgFile}: No label found in CSV")
            continue
        
        label_str = row["label"].values[0]
        label_int = 1 if label_str.lower() == "male" else 0
        
        # Store
        X.append(image)
        y.append(label_int)
        loaded_filenames.append(imgFile)
        
    
    except Exception as e:
        print(f"Error loading {imgFile}: {e}")
 
# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

 
print(f"\n Successfully loaded {len(X)} images")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"    Female (0): {np.sum(y == 0)}")
print(f"    Male (1): {np.sum(y == 1)}")

#Splitting dataset into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {xTrain.shape},  Test: {xTest.shape}")




#Building my model architecture
inPut = Input(shape=(30, 30, 3))         

x = Conv2D(16, (3, 3), activation="relu")(inPut)
x = Conv2D(32, (3, 3), activation="relu")(x)
x = MaxPooling2D()(x)
x = Conv2D(32, (3, 3), activation="relu")(x)
x = Conv2D(64, (3, 3), activation="relu")(x)
x = MaxPooling2D()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(32, activation="relu")(x)
x = Dense(16, activation="relu")(x)


outPut = Dense(units=1, activation="sigmoid", name="output")(x)
model = Model(inputs=inPut, outputs=outPut)
model.summary()

#Compiling my model with an optimizer and loss function
model.compile( optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Model Training on my training data and introducing early stopping to prevent overfitting
history = model.fit(xTrain, yTrain, epochs=20, validation_split=0.2, verbose=1)

# # Model evaluation 
model.evaluate(xTest[:10], yTest[:10])

# Testing model on a few samples
proba   = model.predict(xTest[:5])
y_pred = [1 if p >= 0.5 else 0 for p in proba.flatten()]
label_map = {0: "female", 1: "male"}

print("\nSample predictions:")
for i, pred in enumerate(y_pred):
    print(f"Image {i+1}: predicted = {label_map[pred]},  actual = {label_map[yTest[i]]}")

img = cv2.imread(r"C:\Users\USER\Desktop\RDA 212_Exam\male1.jpg")
img = cv2.resize(img, (30,30))
img = img / 255.0
img = np.expand_dims(img, axis=0)

proba = model.predict(img)
y_pred = 1 if proba[0][0] >= 0.5 else 0
print(f"Predicted gender: {label_map[y_pred]}")