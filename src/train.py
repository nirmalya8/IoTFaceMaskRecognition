from dataset import create_dataset
from utils import get_checkpoint
from model import get_compiled_model

def train_model():
    train_generator,validation_generator = create_dataset()
    model = get_compiled_model()
    checkpoint = get_checkpoint()
    history = model.fit_generator(train_generator,
                              epochs=10,
                              validation_data=validation_generator,
                              callbacks=[checkpoint])
    model.save('final_model.h5')
if __name__=='__main__':
    train_model()