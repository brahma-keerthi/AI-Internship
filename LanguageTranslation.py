import string
import re
from numpy import array,argmax,random,take
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional, RepeatVector, TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras import models    
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt
%matplotlib inline
pd.set_option('display.max_colwidth',200)

#function to read raw text file
def read_text(filename):
  #open the file
  file = open(filename, mode='rt', encoding = 'utf-8')
  #read all text
  text = file.read()
  file.close()
  return text

#split a text into sentences
def to_lines(text):
  sents = text.strip().split('\n')
  sents= [i.split('\t')for i in sents]
  return sents

data = read_text("deu.txt")
deu_eng = to_lines(data)
deu_eng = array(deu_eng)

deu_eng = deu_eng[:50000,:]
#let's take a look at our data
deu_eng

# empty lists
eng_1=[]
deu_1=[]
# populate the list with sentence lengths
for i in deu_eng[:,0]:
  eng_1.append(len(i.split()))
  
for i in deu_eng[:,1]:
    deu_1.append(len(i.split()))

length_df = pd.DataFrame({'eng':eng_1, 'deu':deu_1})
length_df.hist(bins=30)
plt.show()

# Function to build a tokenizer
def tokenization(lines):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(lines)
  return tokenizer

#preare english tokenizer
eng_tokenizer = tokenization(deu_eng[:,0])
eng_vocab_size = len(eng_tokenizer.word_index)+1

eng_length = 8
print('English Vocabulary Size: %d'% eng_vocab_size)

#prepare Deutch tokenizer
deu_tokenizer = tokenization(deu_eng[:,1])
deu_vocab_size = len(deu_tokenizer.word_index)+1

deu_length = 8
print('Deutch Vocabulary Size: %d'% deu_vocab_size)

#encode and pad sequence
def encode_sequences(tokenizer, length, lines):
  #integer encode sequence
  seq = tokenizer.texts_to_sequences(lines)
  #pad sequence with 0 values
  seq = pad_sequences(seq, maxlen = length, padding='post')
  return seq

from sklearn.model_selection import train_test_split
train, test  = train_test_split(deu_eng, test_size=0.2, random_state = 12)

#prepare training data
trainX = encode_sequences(deu_tokenizer, deu_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:,0])

#prepare validation data
testX = encode_sequences(deu_tokenizer, deu_length, train[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, train[:,0])

# build NMT model
def build_model(in_vocab, out_vocab, in_timesteps, out_timesteps, units):
  model = Sequential()
  model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
  model.add(LSTM(units))
  model.add(RepeatVector(out_timesteps))
  model.add(LSTM(units, return_sequences=True))
  model.add(Dense(out_vocab, activation='softmax'))
  return model

  model = build_model(deu_vocab_size, eng_vocab_size, deu_length, eng_length, 512)
rms = optimizers.RMSprop(lr=0.001)
#model.compile(loss='sparse_categorial_crossentropy', optimizers=rms)
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy', metrics=None)

filename = 'model.Kunal'
Checkpoint = ModelCheckpoint(filename, monitor= 'val_loss', verbose=1, save_best_only=True, mode='min')
history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
                   epochs=5, batch_size=512,
                   validation_split = 0.5,
                   callbacks=[Checkpoint], verbose=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
plt.show()

model = load_model('model.Kunal')
preds = model.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))

def get_word(n, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == n:
      return word
      return None

#convert predictions into text (english)
preds_text = []
for i in preds:
  temp = []
  for j in range(len(i)):
    t=get_word(i[j], eng_tokenizer)
    if j>0:
      if (t == get_word(i[j-1], eng_tokenizer)) or (t== None):
        temp.append('')
      else:
        temp.append(t)
          
    else:
      if(t==None):
        temp.append('')
      else:
        temp.append(t)

        preds_text.append(''.join(temp))

pred_df = pd.DataFrame({'actual': test[:,0], 'predicted': preds_text})
pd.set_option('display.max_clowidth', 200)