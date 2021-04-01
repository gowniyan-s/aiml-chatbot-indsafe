from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import re
from build_model import AccidentModel
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)
api = Api(app)

model = AccidentModel()

#Uncomment below line if any changes in model building
#model.build_model() 
m = load_model('model.h5')
tokenizer = Tokenizer(num_words = 3314)


parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictSentiment(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        greetings=['Hello','hello','HI','hi','Hi']

        if any(word in user_query for word in greetings):
            return {'class':'none','response':"Hello, Welcome to Accident Management Portal, Please provide Description:"}

        statement = user_query.lower()
        statement = model.replace_words(statement)
        statement = model.remove_punctuation(statement)
        statement = model.lem(statement)
        statement = re.sub(' +', ' ', statement)

        headline = tokenizer.texts_to_sequences(statement)
        headline = pad_sequences(headline, maxlen = 3314, dtype = 'int32', value = 0)

        sentiment = m.predict(headline)

        print(sentiment)

        print(np.argmax(sentiment))

        # create JSON object
        output = {'predictionGroup': np.argmax(sentiment).item()}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')


if __name__ == '__main__':
    app.run(debug=True)
