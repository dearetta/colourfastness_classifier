#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# In[5]:


glcm_df = pd.read_csv("v4_dataset.csv")

glcm_df.head(20)


# In[6]:


label_distr = glcm_df['label'].value_counts()

label_name = ['G3','G34', 'G4', 'G45','G5']

plt.figure(figsize=(10,10))

my_circle = plt.Circle( (0,0), 0.7, color='white')
plt.pie(label_distr, 
        labels=label_name,  
        autopct='%1.1f%%')

p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# In[7]:


print(label_distr)


# In[8]:


from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def decimal_scaling(data):
    data = np.array(data, dtype=np.float32)
    max_row = data.max(axis=0)
    c = np.array([len(str(int(number))) for number in np.abs(max_row)])
    return data/(10**c)

X = decimal_scaling(glcm_df[['correlation_0', 'correlation_45', 'correlation_90', 'correlation_135',
                             'homogeneity_0', 'homogeneity_45', 'homogeneity_90', 'homogeneity_135',
                             'contrast_0', 'contrast_45', 'contrast_90', 'contrast_135']].values)


# In[9]:


le = LabelEncoder()
le.fit(glcm_df["label"].values)


print(" categorical label : \n", le.classes_)

Y = le.transform(glcm_df['label'].values)
Y = to_categorical(Y)

print("\n\n one hot encoding for sample 0 : \n", Y[0])


# In[10]:


X_train, X_test, y_train, y_test =                     train_test_split(X, 
                                     Y, 
                                     test_size=0.25, 
                                     random_state=42)
  
print("Dimensi data :\n")
print("X train \t X test \t Y train \t Y test")  
print("%s \t %s \t %s \t %s" % (X_train.shape, X_test.shape, y_train.shape, y_test.shape))


# In[11]:


from keras.models import Sequential
from keras.layers import Dense, Activation

import keras
from keras import backend as K
from keras import regularizers

# --------------------- create custom metric evaluation ---------------------
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
  

# --------------------------- create model -------------------------------
def nn_model(max_len):
    
    model = Sequential()
    model.add(Dense(64, 
                    activation="relu",
                    input_shape=(max_len,)))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(5))
    model.add(Activation("softmax"))
    
    model.summary()
    
    optimizer = keras.optimizers.Adam(learning_rate=0.005)
    
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics = ['accuracy', precision, recall])

    return model
 

# ------------------------- check model -----------------------------
def check_model(model_, x, y, x_val, y_val, epochs_, batch_size_):
    
    #x is the input train data, y is the labels

    hist = model_.fit(x, 
                      y,
                      epochs=epochs_,
                      batch_size=batch_size_,
                      validation_data=(x_val,y_val))
    return hist


# In[16]:


max_len = X_train.shape[1]  
print(max_len)

EPOCHS = 300
BATCH_SIZE = 32

model = nn_model(max_len)
history=check_model(model, X_train,y_train,X_test,y_test, EPOCHS, BATCH_SIZE)


# In[17]:


def evaluate_model_(history):
    names = [['accuracy', 'val_accuracy'], 
             ['loss', 'val_loss'], 
             ['precision', 'val_precision'], 
             ['recall', 'val_recall']]
    for name in names :
        fig1, ax_acc = plt.subplots()
        plt.plot(history.history[name[0]])
        plt.plot(history.history[name[1]])
        plt.xlabel('Epoch')
        plt.ylabel(name[0])
        plt.title('Model - ' + name[0])
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.show()
        
evaluate_model_(history)


# In[18]:


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
 # predict test data
y_pred=model.predict(X_test)


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, 
                      classes=['G3','G34', 'G4','G45', 'G5'],
                      normalize=False,
                      title='Confusion matrix, with normalization')


# In[19]:


print(classification_report(y_test.argmax(axis=1), 
                            y_pred.argmax(axis=1), 
                            target_names=['G3','G34', 'G4', 'G45','G5']))


# In[20]:


glcm_df = pd.read_csv("v2_test.csv")

print(glcm_df.shape)

glcm_df.head(45)


# In[21]:


X_new = decimal_scaling(glcm_df[['correlation_0', 'correlation_45', 'correlation_90', 'correlation_135',
                             'homogeneity_0', 'homogeneity_45', 'homogeneity_90', 'homogeneity_135',
                             'contrast_0', 'contrast_45', 'contrast_90', 'contrast_135']].values)

label_name = ['G3','G34', 'G4', 'G45','G5']
Y_new = model.predict_classes(X_new)
# print(X_new.shape)

# show the inputs and predicted outputs
# print("X=%s, Predicted=%s" % (X_new[0], Y_new[0]))
i = 0
num = len(Y_new)
for i in range(num):
    print(i,label_name[int(Y_new[i])])
    i=+1


# In[22]:


X_new = pd.DataFrame(X_new)

X_new.to_csv("Result_v2.csv")
print(X_new.shape)

X_new.head(10)


# In[ ]:





# In[ ]:





# In[ ]:




