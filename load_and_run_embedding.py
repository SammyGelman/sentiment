# Suppress TensorFlow output by setting the environment variable
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('FATAL')

import json
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import io
import os
import pandas as pd

# Suppress TensorFlow output by setting the environment variable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('FATAL')

sent = input("Write a sentence we can check how sarcastic it is: ")

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


#Load list of embeddings from tsv files saved from past training
#Check to see that files exist
vec_file = [s for s in os.listdir() if s.startswith("vecs.tsv")]
meta_file = [s for s in os.listdir() if s.startswith("meta.tsv")]
embeddings_index = {}
if len(vec_file) != 0 and len(meta_file) != 0:
    vecs = []
    with open(vec_file[0]) as f:
        for line in f:
            vec = line.split('\t')
            vec[-1] = vec[-1].strip()
            vecs.append(vec)
    words = []
    with open(meta_file[0]) as f:
        for line in f:
            word = line.split('\n')
            words.append(word[0])
else:
    print("Missing pre_trained embeddings!")
    exit()

for i in range(len(vecs)):
    embeddings_index[words[i]] = vecs[i]

#Here we are going to load the pretrained embedding and 
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#Define an embedding layer which incorporates the embedding matrix. 
pretrained_embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                                       embedding_dim,
                                                       embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                                       input_length=max_length,
)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,
                              embedding_dim,
                              input_length=max_length,
                              ),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.load_weights("weights/cp.ckpt")


new_model = tf.keras.Sequential([pretrained_embedding_layer] + model.layers[1:])

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

num_epochs = 30

# Compile and train the model
new_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
initial_epoch = 0

e = model.layers[0]
weights = e.get_weights()[0]

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# sentence = ["Well that was a fantastic idea, wish I thought of it myself","granny starting to fear spiders in the garden might be real","this is meant to be a non-sarcastic comment", "game of thrones season finale showing this sunday night"]
# sequences = tokenizer.texts_to_sequences(sentence)

sequences = tokenizer.texts_to_sequences([sent])
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
if new_model.predict(padded)[0][0] >= 0.5:
    print("That was probably sarcastic")
else:
    print("Didn't seem sarcastic to me!")
