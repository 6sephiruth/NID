
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import pickle

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

def pre_dataset():

    x_train = pickle.load(open(f'./dataset/kdd/x_train','rb'))
    x_test = pickle.load(open(f'./dataset/kdd/x_test','rb'))
    y_train = pickle.load(open(f'./dataset/kdd/y_train','rb'))
    y_test = pickle.load(open(f'./dataset/kdd/y_test','rb'))
    
    return x_train, x_test, y_train, y_test

def pre_dataset2():

    cols="""duration,
    protocol_type,
    service,
    flag,
    src_bytes,
    dst_bytes,
    land,
    wrong_fragment,
    urgent,
    hot,
    num_failed_logins,
    logged_in,
    num_compromised,
    root_shell,
    su_attempted,
    num_root,
    num_file_creations,
    num_shells,
    num_access_files,
    num_outbound_cmds,
    is_host_login,
    is_guest_login,
    count,
    srv_count,
    serror_rate,
    srv_serror_rate,
    rerror_rate,
    srv_rerror_rate,
    same_srv_rate,
    diff_srv_rate,
    srv_diff_host_rate,
    dst_host_count,
    dst_host_srv_count,
    dst_host_same_srv_rate,
    dst_host_diff_srv_rate,
    dst_host_same_src_port_rate,
    dst_host_srv_diff_host_rate,
    dst_host_serror_rate,
    dst_host_srv_serror_rate,
    dst_host_rerror_rate,
    dst_host_srv_rerror_rate"""

    columns=[]
    
    for c in cols.split(','):
        if(c.strip()):
            columns.append(c.strip())

    columns.append('target')

    attacks_types = {
        'normal': 'normal',
    'back': 'dos',
    'buffer_overflow': 'u2r',
    'ftp_write': 'r2l',
    'guess_passwd': 'r2l',
    'imap': 'r2l',
    'ipsweep': 'probe',
    'land': 'dos',
    'loadmodule': 'u2r',
    'multihop': 'r2l',
    'neptune': 'dos',
    'nmap': 'probe',
    'perl': 'u2r',
    'phf': 'r2l',
    'pod': 'dos',
    'portsweep': 'probe',
    'rootkit': 'u2r',
    'satan': 'probe',
    'smurf': 'dos',
    'spy': 'r2l',
    'teardrop': 'dos',
    'warezclient': 'r2l',
    'warezmaster': 'r2l',
    }

    path = "./dataset/kdd/kddcup.data_10_percent.gz"
    df = pd.read_csv(path,names=columns)

    #Adding Attack Type column
    df['Attack Type'] = df.target.apply(lambda r:attacks_types[r[:-1]])
    #protocol_type feature mapping
    pmap = {'icmp':0,'tcp':1,'udp':2}
    df['protocol_type'] = df['protocol_type'].map(pmap)
    #flag feature mapping
    fmap = {'SF':0,'S0':1,'REJ':2,'RSTR':3,'RSTO':4,'SH':5 ,'S1':6 ,'S2':7,'RSTOS0':8,'S3':9 ,'OTH':10}
    df['flag'] = df['flag'].map(fmap)
    drop_cols = ['service', 'target']
    df = df.drop(drop_cols, axis=1)
    #flag feature mapping
    attack_type = {'normal':0, 'dos':1, 'probe':1, 'r2l':1, 'u2r':1}
    df['Attack Type'] = df['Attack Type'].map(attack_type)

    X, Y = df.iloc[:,:-1].values, df.iloc[:,-1].values

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    pickle.dump(x_train, open(f'./dataset/kdd/x_train','wb'))
    pickle.dump(x_test, open(f'./dataset/kdd/x_test','wb'))
    pickle.dump(y_train, open(f'./dataset/kdd/y_train','wb'))
    pickle.dump(y_test, open(f'./dataset/kdd/y_test','wb'))

def pre_model():

    x_train, x_test, y_train, y_test = pre_dataset()
    
    checkpoint_path = f'model/cp'

    if os.path.exists(f'./model/cp/saved_model.pb'):
        model = tf.keras.models.load_model(checkpoint_path)
        
    else:

        checkpoint = ModelCheckpoint(checkpoint_path, 
                                    save_best_only=True, 
                                    save_weights_only=True, 
                                    monitor='val_loss',
                                    verbose=1)


        model = Sequential([

        Dense(128,input_dim =40,activation = 'relu',kernel_initializer='random_uniform'),
        Dense(64,activation='sigmoid',kernel_initializer='random_uniform'),
        Dense(32,activation='sigmoid',kernel_initializer='random_uniform'),
        Dense(2,activation='softmax')
        ])

        model.compile(loss ='sparse_categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

        model.fit(x_train, y_train, validation_data=(x_test, y_test) , shuffle=True, epochs=10, verbose=1, batch_size=64, callbacks=[checkpoint],)

        model.trainable = False
        model.save('./model/cp')

    return model 

def exists(pathname):
    return os.path.exists(pathname)


from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses


class AnomalyDetector(Model):
    
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            
            layers.Dense(128, activation="relu", input_shape=(40,)),
            layers.Dense(64, activation="relu"),            
            layers.Dense(32, activation="relu"),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(64, activation="relu"),                        
            layers.Dense(128, activation="relu"),
            layers.Dense(40, activation="sigmoid"),
        
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
