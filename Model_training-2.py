#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,Conv2D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, History
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import keras
from keras.models import Sequential, load_model
from keras.losses import MeanSquaredError
from keras.callbacks import ReduceLROnPlateau
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# In[2]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense,Input, Embedding,LSTM,Dropout,Conv1D, MaxPooling1D, GlobalMaxPooling1D,Dropout,Bidirectional,Flatten,BatchNormalization


# In[75]:


import pickle

with open('TF-IDF_glove_weights_train_dataframe.pickle', 'rb') as file:
    train_dataframe = pickle.load(file)


# In[76]:


with open('TF-IDF_glove_weights_test_dataframe.pickle', 'rb') as file:
    test_dataframe = pickle.load(file)


# In[85]:


test_dataframe.head()


# In[21]:


test_dataframe.describe()


# In[77]:


train_dataframe=train_dataframe.sample(frac=1.0, random_state=42)
train_dataframe = train_dataframe.reset_index(drop=True)


# In[78]:


test_dataframe=test_dataframe.sample(frac=1.0, random_state=42)
test_dataframe = test_dataframe.reset_index(drop=True)


# In[79]:


X_train=train_dataframe.iloc[:, :-1].values
y_train=train_dataframe.iloc[:, -1].values


# In[80]:


X_test=test_dataframe.iloc[:, :-1].values
y_test=test_dataframe.iloc[:, -1].values


# ## LSTM model

# In[9]:


get_ipython().run_cell_magic('time', '', "    \n#X_train = X_train.reshape((X_train.shape[0], 200, 1))\n#X_test = X_test.reshape((X_test.shape[0], 200, 1))\nmodel = Sequential()\nmodel.add(LSTM(64, input_shape=(200,1),return_sequences=True))\nmodel.add(Dropout(0.2))\nmodel.add(LSTM(64))\nmodel.add(Dropout(0.2))\nmodel.add(Dense(32, activation='relu'))\nmodel.add(Dropout(0.2))\nmodel.add(Dense(1, activation='sigmoid'))\nmodel.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])\nmodel.summary()\n")


# In[10]:


filepath = 'lstm_best.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
callbacks = [checkpoint]

history = model.fit(X_train, y_train,
                  batch_size=128,
                  epochs=100,
                  validation_split=0.2,
                  callbacks=callbacks)


# ### Loss vs accuracy graph

# In[52]:


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.savefig('model_training_history')
plt.show()


# In[61]:


# Test data
model_lstm = load_model('lstm_best.hdf5')
loss, accuracy = model_lstm.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# ### Confusion matrix

# In[63]:


ypred = model_lstm.predict(X_test)
ypred = (ypred>0.5).astype(int)

cf_matrix = confusion_matrix(y_test, ypred)
classes = unique_labels(y_test, ypred)

fig, ax = plt.subplots()
im = ax.imshow(cf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(cf_matrix.shape[1]),
       yticks=np.arange(cf_matrix.shape[0]),
       xticklabels=classes, yticklabels=classes,
       title='Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

thresh = cf_matrix.max() / 2.
for i in range(cf_matrix.shape[0]):
    for j in range(cf_matrix.shape[1]):
        ax.text(j, i, format(cf_matrix[i, j], 'd'),
                ha="center", va="center",
                color="white" if cf_matrix[i, j] > thresh else "black")
plt.show()


# In[60]:


predictions = model_lstm.predict(X_test)
binary_predictions = (predictions > 0.5).astype(int)
cm = confusion_matrix(y_test, binary_predictions)
print("Confusion Matrix:")
print(cm)


# ## Bi-LSTM model

# In[85]:


get_ipython().run_cell_magic('time', '', "    \n#X_train = X_train.reshape((X_train.shape[0], 200, 1))\n#X_test = X_test.reshape((X_test.shape[0], 200, 1))\nmodel = Sequential()\nmodel.add(Bidirectional(LSTM(50), input_shape=(X_train.shape[1], 1)))\nmodel.add(Dropout(0.2))\nmodel.add(Dense(32, activation='relu'))\nmodel.add(Dropout(0.2))\nmodel.add(Dense(1, activation='sigmoid'))\nmodel.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])\n#model.fit(X_train, y_train, epochs=50, batch_size=64)\nmodel.summary()\n")


# In[87]:


filepath = 'Bi-lstm_best.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
callbacks = [checkpoint]

history = model.fit(X_train, y_train,
                  batch_size=128,
                  epochs=100,
                  validation_split=0.2,
                  callbacks=callbacks)


# ### Test accuracy

# In[67]:


model_bilstm = load_model('Bi-lstm_best.hdf5')
loss, accuracy = model_bilstm.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# ### Confusion matrix

# In[66]:


ypred = model_bilstm.predict(X_test)
ypred = (ypred>0.5).astype(int)

cf_matrix = confusion_matrix(y_test, ypred)
classes = unique_labels(y_test, ypred)

fig, ax = plt.subplots()
im = ax.imshow(cf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(cf_matrix.shape[1]),
       yticks=np.arange(cf_matrix.shape[0]),
       xticklabels=classes, yticklabels=classes,
       title='Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

thresh = cf_matrix.max() / 2.
for i in range(cf_matrix.shape[0]):
    for j in range(cf_matrix.shape[1]):
        ax.text(j, i, format(cf_matrix[i, j], 'd'),
                ha="center", va="center",
                color="white" if cf_matrix[i, j] > thresh else "black")
plt.show()


# ### Loss vs accuracy graph

# In[91]:


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.savefig('model_training_history')
plt.show()


# In[65]:


predictions = model_bilstm.predict(X_test)
binary_predictions = (predictions > 0.5).astype(int)
cm = confusion_matrix(y_test, binary_predictions)
print("Confusion Matrix:")
print(cm)


# ## CNN with LSTM

# In[70]:


model = Sequential()

# Convolutional layer 1
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(200, 1)))
model.add(MaxPooling1D(pool_size=2))

# Convolutional layer 2
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Reshape((int(model.output_shape[1] / 64), 64)))

# LSTM layer
model.add(LSTM(50))

# Dense layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # For binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# In[71]:


# Train the model
filepath = 'CNN_lstm_model.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
callbacks = [checkpoint]

history = model.fit(X_train, y_train,
                  batch_size=64,
                  epochs=100,
                  validation_split=0.2,
                  callbacks=callbacks)


# ### Test accuracy

# In[34]:


modelcnn_lstm = load_model('CNN_lstm_model.hdf5')
loss, accuracy = modelcnn_lstm.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# ### Loss vs accuracy graph

# In[73]:


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.savefig('model_training_history')
plt.show()


# ### Confusion matrix

# In[68]:


ypred = modelcnn_lstm.predict(X_test)
ypred = (ypred>0.5).astype(int)

cf_matrix = confusion_matrix(y_test, ypred)
classes = unique_labels(y_test, ypred)

fig, ax = plt.subplots()
im = ax.imshow(cf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(cf_matrix.shape[1]),
       yticks=np.arange(cf_matrix.shape[0]),
       xticklabels=classes, yticklabels=classes,
       title='Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

thresh = cf_matrix.max() / 2.
for i in range(cf_matrix.shape[0]):
    for j in range(cf_matrix.shape[1]):
        ax.text(j, i, format(cf_matrix[i, j], 'd'),
                ha="center", va="center",
                color="white" if cf_matrix[i, j] > thresh else "black")
plt.show()


# ## CNN Model

# In[81]:


model = Sequential()

model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.5))  

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))  

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# In[82]:


# Train the model
filepath = 'CNN_model.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
callbacks = [checkpoint]

history = model.fit(X_train, y_train,
                  batch_size=32,
                  epochs=100,
                  validation_split=0.2,
                  callbacks=callbacks)
#model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)


# ### Test accuracy

# In[83]:


# Test data accuracy
modelcnn = load_model('CNN_model.hdf5')
loss, accuracy = modelcnn.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# ### Loss vs accuracy graph

# In[84]:


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.savefig('model_training_history')
plt.show()


# ### Confusion matrix

# In[69]:


ypred = modelcnn.predict(X_test)
ypred = (ypred>0.5).astype(int)

cf_matrix = confusion_matrix(y_test, ypred)
classes = unique_labels(y_test, ypred)

fig, ax = plt.subplots()
im = ax.imshow(cf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(cf_matrix.shape[1]),
       yticks=np.arange(cf_matrix.shape[0]),
       xticklabels=classes, yticklabels=classes,
       title='Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

thresh = cf_matrix.max() / 2.
for i in range(cf_matrix.shape[0]):
    for j in range(cf_matrix.shape[1]):
        ax.text(j, i, format(cf_matrix[i, j], 'd'),
                ha="center", va="center",
                color="white" if cf_matrix[i, j] > thresh else "black")
plt.show()


# ## BERT

# In[21]:


import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# In[16]:


with open('Cleaned_train_sentences.pickle', 'rb') as file:
    train_dataframe = pickle.load(file)


# In[17]:


with open('Cleaned_test_sentences.pickle', 'rb') as file:
    test_dataframe = pickle.load(file)


# In[20]:


test_dataframe


# In[22]:


train_data = train_dataframe
test_data = test_dataframe

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize the data
def tokenize_data(data, max_length=128):
    input_ids = []
    attention_masks = []

    for sentence in data['cleaned_content']:
        encoded_data = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids.append(encoded_data['input_ids'])
        attention_masks.append(encoded_data['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(data['class'].values)

    return input_ids, attention_masks, labels

train_input_ids, train_attention_masks, train_labels = tokenize_data(train_data)
test_input_ids, test_attention_masks, test_labels = tokenize_data(test_data)

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_function = torch.nn.CrossEntropyLoss()


# In[25]:


epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
        optimizer.zero_grad()

        input_ids, attention_masks, labels = batch
        outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    accuracy = accuracy = total_correct / total_samples
    print(f'Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_loss},Training Accuracy: {accuracy:.4f}')


# In[26]:


model.eval()
all_predictions = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc='Evaluating'):
        input_ids, attention_masks, labels = batch
        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy())

# Accuraccy of test set
accuracy = accuracy_score(test_labels, all_predictions)
print(f'Test Accuracy: {accuracy:.4f}')

