from keras.preprocessing.image import ImageDataGenerator

def create_dataset():
    TRAINING_DIR = "Data\\Dataset\\train\\train"
    train_datagen = ImageDataGenerator(rescale=1.0/255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
                                                        batch_size=10, 
                                                        target_size=(150, 150))
    VALIDATION_DIR = "Data\\Dataset\\test\\test"
    validation_datagen = ImageDataGenerator(rescale=1.0/255)
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                            batch_size=10, 
                                                            target_size=(150, 150))
    return train_generator, validation_generator
    