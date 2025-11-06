# TAuberculosis-detection-CNN-Classification-
# Tuberculosis Detection from Chest X-rays
# Author: Sameer Mishra
# IIT Madras - Data Science & Applications

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt

# Paths
train_dir = '/content/drive/MyDrive/TB/train'
val_dir = '/content/drive/MyDrive/TB/val'

# Data preprocessing
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=10, zoom_range=0.1, horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(224,224), batch_size=16, class_mode='binary')
val_data = val_gen.flow_from_directory(val_dir, target_size=(224,224), batch_size=16, class_mode='binary')

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    BatchNormalization(),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(train_data, validation_data=val_data, epochs=15)

# Plot
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Tuberculosis Detection Performance")
plt.show()

# Save model
model.save('/content/tb_detection_model.h5')
