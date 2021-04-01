from model import NLPModel
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import random
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from random import randint
import re
import nltk
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
from nltk.corpus import stopwords
#nltk.download('stopwords')
#stopwords.words('english')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet'); 
nltk.download('stopwords')
from pathlib import Path



class AccidentModel:
    def build_model(self):

        # filename = os.path.join(
        #     os.path.dirname(__file__), 'chalicelib', 'all/train.tsv')
        
        #with open('A:\ML\ind-safety\lib\data\IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv') as f:
        data = pd.read_csv('A:\ML\ind-safety\lib\data\IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv')
        
        data['Accident Severity'] = data['Accident Level'].apply(self.create_label)

        data.rename(columns = {'Data': 'Date', 'Genre': 'Gender'}, inplace = True)
        data = data.drop("Unnamed: 0",axis=1)
        data['cleaned_Description'] = data['Description'].apply(lambda x : x.lower())
        print('Replacing apostrophes to the standard lexicons')
        data['cleaned_Description'] = data['cleaned_Description'].apply(lambda x : self.replace_words(x))
        print('Removing punctuations')
        data['cleaned_Description'] = data['cleaned_Description'].apply(lambda x: self.remove_punctuation(x))
        print('Applying Lemmatizer')
        data['cleaned_Description'] = data['cleaned_Description'].apply(lambda x: self.lem(x))
        print('Removing multiple spaces between words')
        data['cleaned_Description'] = data['cleaned_Description'].apply(lambda x: re.sub(' +', ' ', x))
        print(data.shape)
        print('\*'*30)
        max_features = 3314
        maxlen = 30
        embedding_size = 300
        print(data['cleaned_Description'].head())
        tokenizer = Tokenizer(num_words=max_features,lower=True)
        tokenizer.fit_on_texts(data['cleaned_Description'])
        seq = tokenizer.texts_to_sequences(data['cleaned_Description'])
        padded= pad_sequences(seq,maxlen=maxlen,padding="post")
        X=padded
        print(tokenizer.word_index)
        num_words = len(tokenizer.word_index) + 1
        print(num_words)
        EMBEDDING_FILE = 'A:\ML\ind-safety\lib\data\glove.6B.300d.txt'

        embeddings = {}
        for o in open(EMBEDDING_FILE,encoding="utf8"):
            word = o.split(" ")[0]
            # print(word)
            embd = o.split(" ")[1:]
            embd = np.asarray(embd, dtype='float32')
            # print(embd)
            embeddings[word] = embd

        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((num_words, 300))

        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        print(embedding_matrix.shape)
        #glove and glove100 and 300,multilayer bidirectional ///
        embedding_vector_features=100
        model=Sequential()
        model.add(Embedding(max_features, embedding_size,input_length = maxlen,weights = [embedding_matrix]))
        model.add(Bidirectional(LSTM(100)))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        print(model.summary())
        model.save('model.h5')
        y_acc=data['Accident Level']

        encoder = LabelEncoder()
        encoder.fit(y_acc)
        encoded_Y= encoder.transform(y_acc)
        
        dummy_y= np_utils.to_categorical(encoded_Y)
        X_acc=np.array(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_acc, dummy_y, test_size=0.33, random_state=42,shuffle=True)

        model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=150,batch_size=30)
        print('Model training complete')

        return model;
        

       

    def create_label(self,accident_level):
        level = str(accident_level)
        if level.__eq__("I") or level.__eq__("II"):
            return "low"
        if level.__eq__("III"):
            return "moderate"
        if level.__eq__("IV") or level.__eq__("V"):
            return "high"
        if level.__eq__("VI"):
            return "critical"

    appos = {"ain't": "am not", "aren't": "are not", "can't": "cannot", 
            "can't've": "cannot have", "'cause": "because", 
            "could've": "could have", "couldn't": "could not", 
            "couldn't've": "could not have", "didn't": "did not", 
            "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
            "hadn't've": "had not have", "hasn't": "has not", 
            "haven't": "have not", "he'd": "he would", "he'd've": "he would have", 
            "he'll": "he will", "he'll've": "he will have", 
            "he's": "he is", "how'd": "how did", 
            "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
            "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
            "I'll've": "I will have", "I'm": "I am", "I've": "I have", 
            "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
            "it'll": "it will", "it'll've": "it will have", "it's": "it is", 
            "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
            "might've": "might have", "mightn't": "might not", 
            "mightn't've": "might not have", "must've": "must have", 
            "mustn't": "must not", "mustn't've": "must not have", 
            "needn't": "need not", "needn't've": "need not have",
            "o'clock": "of the clock", "oughtn't": "ought not", 
            "oughtn't've": "ought not have", "shan't": "shall not", 
            "sha'n't": "shall not", "shan't've": "shall not have", 
            "she'd": "she would", "she'd've": "she would have", 
            "she'll": "she will", "she'll've": "she will have",
            "she's": "she is", "should've": "should have", 
            "shouldn't": "should not", "shouldn't've": "should not have", 
            "so've": "so have", "so's": "so is", 
            "that'd": "that had", "that'd've": "that would have", 
            "that's": "that that is", "there'd": "there would", 
            "there'd've": "there would have", "there's": "there is", 
            "they'd": "they would", "they'd've": "they would have", 
            "they'll": "they will", "they'll've": "they will have", 
            "they're": "they are", "they've": "they have", 
            "to've": "to have", "wasn't": "was not", "we'd": "we would", 
            "we'd've": "we would have", "we'll": "we will", 
            "we'll've": "we will have", "we're": "we are", 
            "we've": "we have", "weren't": "were not", 
            "what'll": "what will", "what'll've": "what will have", 
            "what're": "what are", "what's": "what is", 
            "what've": "what have", "when's": "when is", 
            "when've": "when have", "where'd": "where did", 
            "where's": "where is", "where've": "where have", 
            "who'll": "who will", "who'll've": "who will have", 
            "who's": "who is", "who've": "who have", 
            "why's": "why is", "why've": "why have", "will've": "will have", 
            "won't": "will not", "won't've": "will not have",
            "would've": "would have", "wouldn't": "would not", 
            "wouldn't've": "would not have", "y'all": "you all", 
            "y'all'd": "you all would", "y'all'd've": "you all would have", 
            "y'all're": "you all are", "y'all've": "you all have", 
            "you'd": "you would", "you'd've": "you would have",
            "you'll": "you will", "you'll've": "you will have", 
            "you're": "you are", "you've": "you have"}

    # Helper function to replace appos
    def replace_words(self,headline):
        cleaned_headlines = []
        for word in str(headline).split():
            if word.lower() in self.appos.keys():
                cleaned_headlines.append(self.appos[word.lower()])
            else:
                cleaned_headlines.append(word)
        return ' '.join(cleaned_headlines)

    # Helper function to remove punctuations
    # Reference: https://www.programiz.com/python-programming/methods/string/translate

    def remove_punctuation(self,text):
        PUNCT_TO_REMOVE = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' #string.punctuation
        """function to remove the punctuation"""
        return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

    # Helper function to lemmatize

    def lemmatize(self,text):
        lemmatizer = WordNetLemmatizer()
        return ''.join([lemmatizer.lemmatize(word) for word in text])

    # Helper function to remove stopwords

    def remove_stopwords(self,text):
        stoplist = set(stopwords.words('english'))
        stoplist.remove('not')
        """function to remove the stopwords"""
        return " ".join([word for word in str(text).split() if word not in stoplist])


    def lem(self,text):
        lemmatizer = WordNetLemmatizer()
        pos_dict = {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}
        return(' '.join([lemmatizer.lemmatize(w,pos_dict.get(t, wn.NOUN)) for w,t in nltk.pos_tag(text.split())]))

    def remove_alpha_numerics(self,description_txt):
        return re.sub('[^a-zA-Z]',' ',description_txt)

    def to_lower_case(self,description_txt):
        return str(description_txt).lower()


    def remove_stopwords(self,word):
        stop_words = set(stopwords.words('english'))
        return [token for token in word if not token in stop_words]


    def stem_tokens(self,tokens):
        stemmer = PorterStemmer()
        return [stemmer.stem(index) for index in tokens]

    
    def lematize_tokens(self,tokens):
        lemma = WordNetLemmatizer()
        return [lemma.lemmatize(word=w,pos='v') for w in tokens]

    def remove_noise(self,tokens):
        return [i for i in tokens if len(i) > 2]

    def to_string(self,tokens):
        return ' '.join(tokens)


# if __name__ == "__main__":
#     self.build_model()
