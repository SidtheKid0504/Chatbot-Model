import json
import string
import random 
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer 
import tensorflow as tf 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout
nltk.download("punkt")
nltk.download("wordnet")

#Initalizing Lemmatizer
lemmatizer = WordNetLemmatizer()

#Initialize Lists
words = []
classes = []
doc_X = []
doc_y = []

#Read JSON File
with open('intents.json') as json_file:
    data = json.load(json_file)

#Loop Through All Intents
#Tokenize All Patterns And Append To Words
#Append Associated Tag With Associated List

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern) 
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["tag"])

    #Add Tag To Classes If Not There
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

#Lemmatize All Words In Vocab And Convert To Lowercase
#If Words Don't Appear In Punctuation

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

#Sort Vocab And Classes 
words = sorted(set(words))
classes = sorted(set(classes))

#List For Training Data
training = []
out_empty = [0] * len(classes)

#Create Bag Of Words Model
for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    
    #Mark Index Of Class That Current Pattern Is Associated
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1

    #Add One Hot Encoded BOW And Associated Classes To Training
    training.append([bow, output_row])

#Shuffle Data And Convert Into Array
random.shuffle(training)
training = np.array(training, dtype=object)

#Split Features And Labels
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

#Model Parameters
input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 200

#Create Model
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation = "softmax"))

#Compile Model
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy']
)

#Fit Model And Summary
print(model.summary())
model.fit(x=train_X, y=train_y, epochs=epochs, verbose=1)

#Utility Functions
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab): 
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens: 
        for idx, word in enumerate(vocab):
            if word == w: 
                bow[idx] = 1
    return np.array(bow)

def pred_class(text, vocab, labels): 
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
        return return_list

def get_response(intents_list, intents_json): 
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents: 
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

#Save Model
model.save('chatbot_model.h5')

#Chatbot Test
while True:
    message = input("Enter 'stop' To Break Execution: ")
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print(result)

    if message == "stop":
        break
