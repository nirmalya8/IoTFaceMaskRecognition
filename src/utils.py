from keras.callbacks import TensorBoard, ModelCheckpoint

def get_checkpoint():
    checkpoint = ModelCheckpoint('model2-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
    return checkpoint