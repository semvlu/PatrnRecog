import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
from tqdm.notebook import tqdm
import warnings
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

data=pd.DataFrame(columns=['raw_data','duration','digit'])
# change train data path
dir_path= 'filepath/PR/recordings/'
for i in tqdm(os.listdir(dir_path)):
        raw_data,frame_rate=librosa.load(dir_path+i)
        duration=librosa.get_duration(S=raw_data,sr=frame_rate)
        data.loc[len(data.index)]=[raw_data,duration,i.split('_')[0]] # append label as in file name

X_train, X_vld, y_train, y_vld = train_test_split(data[['raw_data','duration']],data['digit'], test_size=0.3, random_state=45,stratify=data['digit'])

X_test = X_vld[0:600]
X_vld = X_vld[600:]
y_test = y_vld[0:600]
y_vld = y_vld[600:]

max_length=20366

X_train_pad=tf.keras.preprocessing.sequence.pad_sequences(X_train['raw_data'],maxlen=max_length, dtype='float32')
X_vld_pad=tf.keras.preprocessing.sequence.pad_sequences(X_vld['raw_data'],maxlen=max_length, dtype='float32')
X_test_pad=tf.keras.preprocessing.sequence.pad_sequences(X_test['raw_data'],maxlen=max_length, dtype='float32')

X_train_mask=np.where(X_train_pad>0.0,True,False)
X_vld_mask=np.where(X_vld_pad>0.0,True,False)
X_test_mask=np.where(X_test_pad>0.0,True,False)


def convert_to_spectrogram(raw_data):
    spect = librosa.feature.melspectrogram(y=raw_data, n_mels=64) # n_mels as output shape
    mel_spect = librosa.power_to_db(S=spect, ref=np.max)
    return mel_spect
X_train_spectrogram=np.array([convert_to_spectrogram(np.array([float(i) for i in X_train_pad[k] ])) for k in range(len(X_train_pad)) ])
X_vld_spectrogram=np.array([convert_to_spectrogram(np.array([float(i) for i in X_vld_pad[k] ])) for k in range(len(X_vld_pad)) ])
X_test_spectrogram=np.array([convert_to_spectrogram(np.array([float(i) for i in X_test_pad[k] ])) for k in range(len(X_test_pad)) ])

input_layer=Input(shape=(64,40), dtype=np.float32,name='input_layer')
lstm=LSTM(500,name='lstm_layer',return_sequences=True)(input_layer)
d1=Dense(120,activation='relu',name='dense1')(tf.math.reduce_mean(lstm, 2))
d2=Dense(60,activation='relu',name='dense2')(d1)
d3=Dense(10,activation='softmax',name='dense3')(d2)
model = Model(inputs=input_layer, outputs=d3)

opt= tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy' )
tf.keras.backend.clear_session()
model.fit(X_train_spectrogram,y_train.astype('int')\
           ,validation_data=(X_vld_spectrogram,y_vld.astype('int'))\
           ,batch_size=32,epochs=100)

pred = model.predict(X_test_spectrogram) # 2D list, 1st sublist:[0.02150052 0.19925891 0.0276884  ... 0.02144623 0.00402112 0.0022369 ]
print(len(model.predict(X_test_spectrogram)[0])) #out: 10

ttlpred = []
for i in pred:
    maxidx = i.argmax()
    ttlpred.append(maxidx)

cm = confusion_matrix(y_test.astype('int'), ttlpred)
print("Accuracy: ", accuracy_score(y_test.astype('int'), ttlpred))
print("Recall: ", recall_score(y_test.astype('int'), ttlpred, average='macro'))
print("F1 Score: ", f1_score(y_test.astype('int'), ttlpred, average='macro'))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()