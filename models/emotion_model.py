import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
# Define the CNN model
def create_emotion_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 classes: angry, disgust, fear, happy, neutral, sad, surprise
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Paths to train and test directories
train_dir = 'data/emotion/train'
val_dir = 'data/emotion/test'

# Data generators
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='sparse'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='sparse'
)

# Initialize and train the model
emotion_model = create_emotion_model()
emotion_model.fit(train_generator, validation_data=val_generator, epochs=20)

# Save the model
model_path = 'models/emotion_model.h5'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
emotion_model.save(model_path)
print(f"Model saved to {model_path}")
