from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_image_generators(data_dir, target_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    return train_generator, validation_generator
