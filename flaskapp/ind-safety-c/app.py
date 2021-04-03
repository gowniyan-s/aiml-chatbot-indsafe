from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import numpy as np
from build_model import AccidentModel
from fasttext import load_model
from gensim.utils import simple_preprocess

app = Flask(__name__)
api = Api(app)

model = AccidentModel()

m = load_model('model_ft.bin')

parser = reqparse.RequestParser()

parser.add_argument('query')


class PredictSentiment(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        print('\*'*30)
        print("query:"+user_query)
        print('\*'*30)
        greetings=['Hello','hello']

        if any(word in user_query for word in greetings):
            return {'class':'none','response':"Hello, Welcome to Accident Management Portal, Please provide Description:"}
       
        print(m.predict(user_query))

        output=m.predict( ' '.join(simple_preprocess(user_query)))[0][0]

        level = str(output).replace("__label__","")

        output = {'potentialAccidentLevel': level}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')


if __name__ == '__main__':
    app.run(debug=True)
