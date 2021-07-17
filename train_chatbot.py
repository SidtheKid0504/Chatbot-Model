#Imports
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import json

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

#Initialize Lists Of Natural Language Data
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

"""
Nested For Loop To Add All Words To words List and Patterns With Their Corresponding
Tags Are Added To documents List
"""
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #Tokenize Each Word
        w = nltk.word_tokenize(pattern)
        words.extend(w)

        #Adding Documents
        documents.append((w, intent['tag']))

        #Adding Classes To classes List
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

        #Lemmatize (Ex: Walked Lemmatized is Walk) and Lowercase All Words
        words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

        #Sort Lists
        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))

        pickle.dump(words, open('words.pkl', 'wb'))
        pickle.dump(classes, open('classes.pkl', 'wb'))

#Initializing Empty Training Data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    #Initializing Bag Of Words
    bag = []

    #List Of Tokenized Words For Pattern
    pattern_words = doc[0]
    #Lemmatize Each Word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # Create Bag Of Words With 1 if Word Match Found In Current Pattern
    for word in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    #Output = '0' For Each Tag And '1' For Current Tag 
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

#Shuffle Features And Convert Into Array
random.shuffle(training)
training = np.array(training)

#Create Training Data(X - Patterns, Y - Intents)
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training Data Is Ready")

#Create Model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Compile Model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])

#Fit And Save Model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

print("Model Created")