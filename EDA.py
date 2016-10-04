import numpy as np
import pandas as pd


import gzip

import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression



class calcReviewQuality( object ):

    def __init__( self, file_name ):
        self.file_name = file_name
        self.stop = set(stopwords.words('english'))
        self.wnl = WordNetLemmatizer()

        self.N_words = []
        self.N_unique_words = []
        self.N_words_test = []
        self.N_unique_words_test = []


        self.model = None
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.y_predict = []

    def parse( self ):
        g = gzip.open(self.file_name, 'r')

        for l in g:
            yield eval(l)
            # print type(l)


    def extract_features( self ):
        pass

    def getData( self ):
        cnt = 0
        for review in self.parse():

            helpfulness = review['helpful']

            text = review['reviewText']
            tokenlizeText = text.split(' ')

            n_words = len(tokenlizeText)
            unique_words = [ word for word in tokenlizeText if word.strip(string.punctuation).lower() not in self.stop]
            n_unique_words = len(unique_words)

            if helpfulness[1] > 0:
                rating = float(helpfulness[0]) / helpfulness[1]

            if helpfulness[1] > 5:
                self.N_words.append( n_words )
                self.N_unique_words.append( n_unique_words )

                ## For Logistic Regression
                # if rating > 0.5:
                #     rating = 1
                # else:
                #     rating = 0
                # # print rating

                self.y_train.append( rating )
            elif helpfulness[1] > 0:  # helpfulness: 0~5
                self.N_words_test.append( n_words )
                self.N_unique_words_test.append( n_unique_words )
                self.y_test.append( rating )
            else:
                print "helpfulness[1] == {}, helpfulness[0]={}".format( helpfulness[1] , helpfulness[0] )


            cnt += 1
            if cnt > 205:
                break

        self.X_train.append( self.N_words )
        self.X_train.append( self.N_unique_words )
        self.X_train = np.array( self.X_train )
        self.X_train = self.X_train.T

        self.X_test.append( self.N_words_test )
        self.X_test.append( self.N_unique_words_test )
        self.X_test = np.array( self.X_test )
        self.X_test = self.X_test.T

        # df = pd.DataFrame( self.X_train, columns = ['N_words', 'N_unique_words'])

    def extract_features( self ):
        pass



    def train_model( self, model = 'LinearRG' ):
        if model == 'LinearRG':
            self.model = LinearRegression()
        elif model == 'LogisticRG':
            self.model = LogisticRegression()
        else:
            print 'Model not supported yet.'

        self.model.fit( self.X_train, self.y_train )
        # The coefficients
        print('Coefficients: \n', self.model.coef_)

    def predict( self ):
        self.y_predict = self.model.predict( self.X_test )


if __name__ == "__main__":
    file_name = 'reviews_Books_5.json.gz'
    crq = calcReviewQuality( file_name )

    crq.getData()
    # crq.extract_features()

    crq.train_model( model = 'LinearRG') #'LogisticRG')#'LogisticRG' )
    crq.predict()
    print crq.y_test, crq.y_predict




    # total number of words,
    # number of unique words,
    # number of paragraphs,
    # number of sentences

        # review['reviewerID']
        # review['reviewText'] #
        # review['overall']  # score
        # review['helpful']  # label






    # 'asin': '000100039X'
