import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Configuration
DATA_DIR = 'mikan_syori'
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = 'mikan_model.h5'

def train_model():
    print("Checking GPU availability...")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Data Augmentation
    # Since the dataset is small, we use heavy augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # Use 20% for validation
    )

    print(f"Loading data from {DATA_DIR}...")
    
    # Load Data
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = len(train_generator.class_indices)
    print(f"Detected {num_classes} classes: {train_generator.class_indices}")

    # Base Model (MobileNetV2)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Freeze base model layers
    base_model.trainable = False

    # Custom Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # Fine-tuning (Optional but recommended for transfer learning)
    print("Fine-tuning...")
    base_model.trainable = True
    # Freeze all layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE/10), # Lower learning rate
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    history_fine = model.fit(
        train_generator,
        epochs=10, # Train for a few more epochs
        validation_data=validation_generator
    )

    # Save Model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Save class indices mapping
    import json
    with open('class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
    print("Class indices saved to class_indices.json")

if __name__ == '__main__':
    train_model()
