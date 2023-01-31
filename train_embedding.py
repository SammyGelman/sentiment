import tensorflow as tf
import pandas as pd
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import os
import io

#load data
with open('sarcasm.json', 'r') as f:
    datastore = json.load(f)

#create lists to store data
sentences = []
labels = []

#seperate data into lists
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

#set training size
percent_training = 0.8
training_size = int(percent_training*len(sentences))

#Split data and labels to training and testing sets
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

#Tokenize data
#Note: We only tokenize data on the training set, it is important that the model knows nothing of the testing data to reflect real world use

#Parameters
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type =  "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences,  
                                maxlen=max_length, padding=padding_type, 
                                truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length,
                                padding=padding_type, truncating=trunc_type)

print("Data is ready")

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#Search and list saved weights
dirs = [s for s in os.listdir() if s.startswith("weights")]
print(dirs, '\n\n\n')

# Save weights to  checkpoint file
checkpoint_path = "weights/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#import useful callback functions 

# Create a callback object that saves the model's weights
checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=True, 
        mode='min')

#Stops training if imporvment is no longer being observed
earlystop = EarlyStopping(monitor='val_loss', patience=12, verbose=1)

#CSVLogger logs epoch, acc, loss, val_acc, val_loss
log_csv = CSVLogger('log.csv',separator=',', append=True)

callbacks_list = [checkpoint, earlystop, log_csv]

print("Model built")

model.summary()

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

num_epochs = 30

# Compile and train the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
initial_epoch = 0

history = model.fit(training_padded, training_labels, epochs=num_epochs,
                validation_data=(testing_padded, testing_labels),
                verbose=2, callbacks=callbacks_list)

e = model.layers[0]
weights = e.get_weights()[0]


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

sentence = ["Well that was a fantastic idea, wish I thought of it myself","granny starting to fear spiders in the garden might be real","this is meant to be a non-sarcastic comment", "game of thrones season finale showing this sunday night"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))
