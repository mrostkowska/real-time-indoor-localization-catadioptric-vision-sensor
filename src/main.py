import tensorflow as tf

from train_model import *
from test_model import *

def main():
    print("START TRAIN")
    train()
    print("FINISH TRAIN")

    print("START TEST")
    test_model()
    print("FINISH TEST")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
# Restrict TensorFlow to only use the first GPU
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        
        main()
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
else:
    print("bez gpu")
    main()