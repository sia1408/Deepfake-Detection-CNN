import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.cnn_model import create_model
import os

def train_model():
    data_dir = 'data/deepfake-faces'
    
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    model = create_model()

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='models/model_checkpoint.h5',
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // 32,
        callbacks=[checkpoint_callback]
    )

    model.save('models/final_model.h5')

if __name__ == "__main__":
    train_model()
