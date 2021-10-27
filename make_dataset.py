import argparse
import os
import yaml
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint

import time


from tqdm import trange

import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# enable memory growth
physical_devices = tf.config.list_physical_devices('GPU')

for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)

os.environ['TF_DETERMINISTIC_OPS'] = '0'

from utils import *
from attack import *
from attribution import *

x_train, x_test, y_train, y_test = pre_dataset()

model = pre_model()


ATTACK_METHOD = 'fgsm'
ATTACK_EPS = 0.05

XAI_METHOD = 'ig'


# if ATTACK_METHOD == 'fgsm':

#     print("FGSM 시작")

#     if exists(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_test') and exists(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_label'):

#         attack_test = pickle.load(open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_test','rb'))
#         attack_label = pickle.load(open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_label','rb'))

#     else:

#         attack_test, attack_label = [], []

#         for i in trange(len(x_test)):
            
#             adv_data = eval('fgsm')(model, x_test[i], ATTACK_EPS) # (28, 28, 1)

#             pred_adv_data = model.predict(tf.expand_dims(adv_data, 0))
#             pred_adv_data = np.argmax(pred_adv_data)

#             if y_test[i] != pred_adv_data:
#                 attack_label.append(1)
#                 attack_test.append(adv_data)

#             else:
#                 attack_label.append(0)
#                 attack_test.append(x_test[i])

#         attack_test, attack_label = np.array(attack_test), np.array(attack_label)

#         pickle.dump(attack_test, open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_test','wb'))
#         pickle.dump(attack_label, open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_label','wb'))

# elif ATTACK_METHOD == 'pgd':
    
#     print("PGD 시작")
    
#     if exists(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_test') and exists(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_label'):
    
#         attack_test = pickle.load(open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_test','rb'))
#         attack_label = pickle.load(open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_label','rb'))

#     else:

#         attack_test, attack_label = [], []

#         for i in trange(len(x_test)):
            
#             adv_data = eval('pgd')(model, x_test[i], ATTACK_EPS) # (28, 28, 1)

#             pred_adv_data = model.predict(tf.expand_dims(adv_data, 0))
#             pred_adv_data = np.argmax(pred_adv_data)

#             if y_test[i] != pred_adv_data:
#                 attack_label.append(1)
#                 attack_test.append(adv_data)

#             else:
#                 attack_label.append(0)
#                 attack_test.append(x_test[i])

#         attack_test, attack_label = np.array(attack_test), np.array(attack_label)

#         pickle.dump(attack_test, open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_test','wb'))
#         pickle.dump(attack_label, open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_label','wb'))


# elif ATTACK_METHOD == 'mim':

#     print("MIM 시작")

#     if exists(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_test') and exists(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_label'):
    
#         attack_test = pickle.load(open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_test','rb'))
#         attack_label = pickle.load(open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_label','rb'))

#     else:
#         attack_test, attack_label = [], []

#         for i in trange(len(x_test)):
            
#             adv_data = eval('mim')(model, x_test[i], ATTACK_EPS) # (28, 28, 1)

#             pred_adv_data = model.predict(tf.expand_dims(adv_data, 0))
#             pred_adv_data = np.argmax(pred_adv_data)

#             if y_test[i] != pred_adv_data:
#                 attack_label.append(1)
#                 attack_test.append(adv_data)

#             else:
#                 attack_label.append(0)
#                 attack_test.append(x_test[i])

#         attack_test, attack_label = np.array(attack_test), np.array(attack_label)

#         pickle.dump(attack_test, open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_test','wb'))
#         pickle.dump(attack_label, open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_label','wb'))

# elif ATTACK_METHOD == 'bim':
    
#     print("BIM 시작")

#     if exists(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_test') and exists(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_label'):
    
#         attack_test = pickle.load(open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_test','rb'))
#         attack_label = pickle.load(open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_label','rb'))

#     else:
#         attack_test, attack_label = [], []

#         for i in trange(len(x_test)):
            
#             adv_data = eval('bim')(model, x_test[i], ATTACK_EPS) # (28, 28, 1)

#             pred_adv_data = model.predict(tf.expand_dims(adv_data, 0))
#             pred_adv_data = np.argmax(pred_adv_data)

#             if y_test[i] != pred_adv_data:
#                 attack_label.append(1)
#                 attack_test.append(adv_data)

#             else:
#                 attack_label.append(0)
#                 attack_test.append(x_test[i])

#         attack_test, attack_label = np.array(attack_test), np.array(attack_label)

#         pickle.dump(attack_test, open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_test','wb'))
#         pickle.dump(attack_label, open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_label','wb'))



# elif ATTACK_METHOD == 'cw':
#     """Return non-targeted cw attack data
    
#     """
#     if exists(f'./dataset/{ATTACK_METHOD}/test') and exists(f'./dataset/{ATTACK_METHOD}/label'):
#         attack_test = pickle.load(open(f'./dataset/{ATTACK_METHOD}/test','rb'))
#         attack_label = pickle.load(open(f'./dataset/{ATTACK_METHOD}/label','rb'))

#     else:
#         attack_test, attack_label = [], []

#         for i in trange(len(x_test)):
            
#             adv_data = eval('cw')(model, x_test[i]) # (28, 28, 1)

#             pred_adv_data = model.predict(tf.expand_dims(adv_data, 0))
#             pred_adv_data = np.argmax(pred_adv_data)

#             if y_test[i] != pred_adv_data:
#                 attack_label.append(1)
#                 attack_test.append(adv_data)

#             else:
#                 attack_label.append(0)
#                 attack_test.append(x_test[i])

#         attack_test, attack_label = np.array(attack_test), np.array(attack_label)

#         pickle.dump(attack_test, open(f'./dataset/{ATTACK_METHOD}/test','wb'))
#         pickle.dump(attack_label, open(f'./dataset/{ATTACK_METHOD}/label','wb'))


# if XAI_METHOD == 'saliency':
#     """Returns extracting saliency maps
    
#     """
#     if exists(f'./dataset/{XAI_METHOD}/train'):
#         g_train = pickle.load(open(f'./dataset/{XAI_METHOD}/train','rb'))

#     else:
#         g_train= []

#         for i in trange(len(x_train)):
#             g_train.append(eval('saliency_map')(model, x_train[i])) # (28, 28, 1)

#         g_train = np.array(g_train)

#         pickle.dump(g_train, open(f'./dataset/{XAI_METHOD}/train','wb'))

# elif XAI_METHOD == 'ig':
#     """Returns extracting ig
    
#     """
#     if exists(f'./dataset/{XAI_METHOD}/train'):
#         ig_train = pickle.load(open(f'./dataset/{XAI_METHOD}/train','rb'))

#     else:
#         print("IG 시작")
#         ig_train = []

#         for i in trange(len(x_train)):
#             ig_train.append(eval('ig')(model, x_train[i])) # (28, 28, 1)

#         ig_train = np.array(ig_train)

#         pickle.dump(ig_train, open(f'./dataset/{XAI_METHOD}/train','wb'))






attack_test = pickle.load(open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_test','rb'))


if XAI_METHOD == 'saliency':
    """Returns extracting saliency maps
    
    """
    if exists(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_{XAI_METHOD}'):
        g_train = pickle.load(open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_{XAI_METHOD}','rb'))

    else:
        g_train= []

        for i in trange(len(attack_test)):
            g_train.append(eval('saliency_map')(model, attack_test[i])) # (28, 28, 1)

        g_train = np.array(g_train)

        pickle.dump(g_train, open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_{XAI_METHOD}','wb'))



elif XAI_METHOD == 'ig':
    """Returns extracting ig
    
    """
    if exists(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_{XAI_METHOD}'):
        ig_train = pickle.load(open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_{XAI_METHOD}','rb'))

    else:
        print("IG 시작")
        ig_train = []

        for i in trange(len(attack_test)):
            ig_train.append(eval('ig')(model, attack_test[i])) # (28, 28, 1)

        ig_train = np.array(ig_train)

        pickle.dump(ig_train, open(f'./dataset/{ATTACK_METHOD}/{ATTACK_EPS}_{XAI_METHOD}','wb'))
