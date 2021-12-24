'''PROGRAM INI MENGGUNAKAN FITUR STFT
YANG DIUBAH DARI MEL SEPCTROGAM'''

import IPython.display as ipd
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.optimizers import SGD

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

#Load the dataset labels
train_data=pd.read_csv(r'C:\Users\MRizkiP\Documents\Source Code\PYTHON\Dataset\Dataset Logat\train\train2.csv')
valid_test_data=pd.read_csv(r'C:\Users\MRizkiP\Documents\Source Code\PYTHON\Dataset\Dataset Logat\valid&test\valid&test.csv')
valid_test_data_shuffled=valid_test_data.iloc[np.random.permutation(len(valid_test_data))]
valid_test_data_shuffled=valid_test_data_shuffled.reset_index(drop=True)
# load data (BUAT UNTUK MENGOLAH SEMUA DATA DI FOLDER TRAIN!)
st=[]
lab=[]

st1=[]
lab1=[]

#Feature Extraction
for i in tqdm(range(len(train_data))):
  wav='C:/Users/MRizkiP/Documents/Source Code/PYTHON/Dataset/Dataset Logat/train/'+str(train_data.ID[i])+'.wav'                                
  X, s_rate = librosa.load(wav, sr=16000)
  l=train_data.Class[i]
  lab.append(l)
  m = np.mean(librosa.feature.melspectrogram(X, sr=s_rate, n_fft=2048, hop_length=64, n_mels=256).T,axis=0)
  S_inv =librosa.feature.inverse.mel_to_stft(m, n_fft=2048)
  st.append(S_inv)

for i in tqdm(range(len(valid_test_data))):
  wav='C:/Users/MRizkiP/Documents/Source Code/PYTHON/Dataset/Dataset Logat/valid&test/'+str(valid_test_data_shuffled.ID[i])+'.wav'                                
  X, s_rate = librosa.load(wav, sr=16000)
  l1=valid_test_data_shuffled.Class[i]
  lab1.append(l1)
  m1 = np.mean(librosa.feature.melspectrogram(X, sr=s_rate, n_fft=2048, hop_length=64, n_mels=256).T,axis=0)
  S_inv1 =librosa.feature.inverse.mel_to_stft(m1, n_fft=2048)
  st1.append(S_inv1)
  
# Saving each feature seperately.
stf = pd.DataFrame(st)
stf.to_csv('C:/Users/MRizkiP/Documents/Source Code/PYTHON/Dataset/Dataset Logat/train/stft.csv', index=False)
la = pd.DataFrame(lab)
la.to_csv('C:/Users/MRizkiP/Documents/Source Code/PYTHON/Dataset/Dataset Logat/train/labels_stft.csv', index=False)
la = pd.get_dummies(lab) #Encode the label

stf1 = pd.DataFrame(st1)
stf1.to_csv('C:/Users/MRizkiP/Documents/Source Code/PYTHON/Dataset/Dataset Logat/valid&test/me_valid&test_random_stft.csv', index=False)
la1 = pd.DataFrame(lab1)
la1.to_csv('C:/Users/MRizkiP/Documents/Source Code/PYTHON/Dataset/Dataset Logat/valid&test/labels_valid&test_random_stft.csv', index=False)
la1 = pd.get_dummies(lab1) #Encode the label

label_columns=la.columns #To get the classes
label_columns1=la1.columns
target = la.to_numpy() #Convert labels to numpy array
target1 = la1.to_numpy()

print(label_columns)
print(label_columns.shape)

#Normalize the feature
tran = StandardScaler()
features_train = tran.fit_transform(stf)
features_train1 = tran.fit_transform(stf1)

#Create train, validation and test dataset
#train dataset
feat_train=features_train
target_train=target

#validation and test dataset
y_train=features_train1[:72]
y_val=target1[:72]
test_data=features_train1[72:]
test_label=np.array(lab1[72:])
print("Training",feat_train.shape)
print(target_train.shape)
print("Validation",y_train.shape)
print(y_val.shape)
print("Test",test_data.shape)
print(test_label.shape)

#Build the model
model = Sequential()

model.add(Dense(1025, input_shape=(1025,), activation = 'relu'))

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.6))

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.6))

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(6, activation = 'softmax'))

opt=SGD()
#opt=Adam()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

history = model.fit(feat_train, target_train, batch_size=108,epochs=5000, validation_data=(y_train, y_val))

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss_acc=history.history['loss']
Val_loss_acc=history.history['val_loss']

# Set figure size.
plt.figure(figsize=(10, 7))

# Generate line plot of training, testing loss over epochs.
plt.plot(train_acc, label='Training Accuracy', color='blue')
plt.plot(val_acc, label='Validation Accuracy', color='yellow')

# Set title
plt.title('Training and Validation Accuracy', fontsize = 21)
plt.xlabel('Epoch', fontsize = 15)
plt.legend(fontsize = 15)
plt.ylabel('Accuracy', fontsize = 15)
plt.show()

plt.plot(loss_acc, label='Loss', color='red')

plt.title('Training Loss', fontsize = 21)
plt.xlabel('Epoch', fontsize = 15)
plt.legend(fontsize = 15)
plt.ylabel('Loss', fontsize = 15)
plt.show()

plt.plot(Val_loss_acc, label='Loss', color='brown')

plt.title('Validation Loss', fontsize = 21)
plt.xlabel('Epoch', fontsize = 18)
plt.legend(fontsize = 18)
plt.ylabel('Loss', fontsize = 18)
plt.show()

# To predict the labels of test data
pred = np.argmax(model.predict(test_data), axis=-1)
predict=pred.astype(int)
print(predict)
print(label_columns1)

prediction=[]
for i in predict:
  j=label_columns1[i]
  prediction.append(j)

print(prediction)

#Menentukan nilai True Positive (k)
k=0
for i, j in zip(test_label,prediction):
    if i==j:
       k=k+1
print(str(k)+' dari '+str(len(test_data))+' label pada data uji terprediksi dengan benar.')


conf_mat=multilabel_confusion_matrix(test_label, prediction, labels=label_columns.to_numpy())
print(conf_mat)

# Accuracy
acc=accuracy_score(test_label, prediction)
# Recall
recall=recall_score(test_label, prediction, average=None,zero_division=1)
# Precision
preci=precision_score(test_label, prediction, average=None,zero_division=1)
#F1-score
F1_score = f1_score(test_label, prediction, average=None, zero_division=1)

print('Accuracy: '+str(acc))
print('Recall: '+str(recall))
print('Precision: '+str(preci))
print('F1 Score: '+str(F1_score))
