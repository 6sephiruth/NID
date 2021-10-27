import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from utils import *
from attack import *

x_train, x_test, y_train, y_test = pre_dataset()

model = pre_model()

ATTACK_METHOD = 'fgsm'
ATTACK_EPS = 0.1

XAI_METHOD = 'saliency'


xai_train = pickle.load(open(f'./dataset/{XAI_METHOD}/train','rb'))
xai_train = np.reshape(xai_train, (len(xai_train), 40))

attack_test = pickle.load(open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_test','rb'))
attack_label = pickle.load(open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_label','rb'))

xai_test = pickle.load(open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_{XAI_METHOD}','rb'))

normal_list = np.where(attack_label == 0)[0]
adversarial_list = np.where(attack_label == 1)[0]

xai_normal_test = attack_test[normal_list][:len(adversarial_list)]
xai_adversarial_test = attack_test[adversarial_list]

xai_normal_test = np.reshape(xai_normal_test, (len(xai_normal_test), 40))
xai_adversarial_test = np.reshape(xai_adversarial_test, (len(xai_adversarial_test), 40))



autoencoder = AnomalyDetector()

autoencoder.compile(optimizer='adam', loss='mae')

checkpoint_path = 'auto/1'

if exists(f'./auto/1/saved_model.pb'):

    model = tf.keras.models.load_model(checkpoint_path)

else:
    checkpoint = ModelCheckpoint(checkpoint_path, 
                                save_best_only=True, 
                                save_weights_only=True, 
                                monitor='val_loss',
                                verbose=1)

    history = autoencoder.fit(xai_train, xai_train, 
            epochs=5,
            batch_size=32,
            validation_data=(xai_normal_test, xai_normal_test),
            shuffle=True,
            callbacks=[checkpoint])

    autoencoder.load_weights(checkpoint_path)
    autoencoder.trainable = False

    model.save('./auto/1')

pred_train = autoencoder.predict(xai_train)

mae_loss = np.mean(np.abs(xai_train, pred_train), axis=0)
mse_loss = np.mean((xai_train - pred_train)**2, axis=0)

#mae_threshold = np.mean(mae_loss) + np.std(mae_loss)
#mse_threshold = np.mean(mse_loss) + np.std(mse_loss)


test_adversarial = autoencoder.predict(tf.expand_dims(xai_adversarial_test[0], 0))
# print(test_adversarial.shape)

