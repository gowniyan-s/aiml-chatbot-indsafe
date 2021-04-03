import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn import metrics
import fasttext
from gensim.utils import simple_preprocess
import csv


class AccidentModel:
    def build_model(self):
        print("Reading file")
        data = pd.read_csv('A:\ML\ind-safety\lib\data\IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv')
        print(data.shape)
        print('\*'*30)
        
        df = data[['Potential Accident Level', 'Description']].copy()
        df.shape 
        print(df.head())                         
        #dataset for training with preprocesed text
        df_ftxt_train, df_ftxt_test, y1, y2 = train_test_split(df,df,test_size=0.1, random_state=1)

        df_ftxt_train.iloc[:, 1] = df_ftxt_train.iloc[:, 1].apply(lambda x: ' '.join(simple_preprocess(x)))
        df_ftxt_test.iloc[:, 1] = df_ftxt_test.iloc[:, 1].apply(lambda x: ' '.join(simple_preprocess(x)))

        # Prefixing each row of the 'Potential Accident Level' column with '__label__'
        df_ftxt_train.iloc[:, 0] = df_ftxt_train.iloc[:, 0].apply(lambda x: '__label__' + x)
        df_ftxt_test.iloc[:, 0] = df_ftxt_test.iloc[:, 0].apply(lambda x: '__label__' + x)
        df_ftxt_train.head()
        
        #saving into text files
        df_ftxt_train[['Potential Accident Level', 'Description']].to_csv('data-fasttext-train.txt', 
                                                index = False, 
                                                sep = ' ',
                                                header = None, 
                                                quoting = csv.QUOTE_NONE, 
                                                quotechar = "", 
                                                escapechar = " ")

        df_ftxt_test[['Potential Accident Level', 'Description']].to_csv('data-fasttext-test.txt', 
                                                index = False, 
                                                sep = ' ',
                                                header = None, 
                                                quoting = csv.QUOTE_NONE, 
                                                quotechar = "", 
                                                escapechar = " ")

        dfu=df_ftxt_test
        print("running model")
        # Training the fastText classifier
        ftxt_clf_gensim = fasttext.train_supervised('data-fasttext-train.txt', wordNgrams = 4, epoch=75, lr=0.5, loss='ova')
        # Save the trained model
        ftxt_clf_gensim.save_model("model_ft.bin")

        # Evaluating performance on the entire test file
        print(ftxt_clf_gensim.test('data-fasttext-test.txt'))

        dfu['pred']=dfu['Description'].map(lambda x: ftxt_clf_gensim.predict(x)[0][0])
        print(dfu.head())
        print('\t\t\t\tCLASSIFICATIION METRICS\n')
        print(metrics.classification_report(dfu['Potential Accident Level'], dfu['pred'],target_names= dfu['Potential Accident Level'].unique()))

if __name__ == "__main__":
    AccidentModel().build_model()
