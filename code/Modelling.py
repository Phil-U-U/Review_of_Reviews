import numpy as np
import pandas as pd


import gzip

import string
#from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error as mse

import matplotlib.pyplot as plt



class calcReviewQuality( object ):

    def __init__( self, file_name ):
        self.file_name = file_name
        self.stop = set(stopwords.words('english'))
        self.wnl = WordNetLemmatizer()

        self.model = None
        self.X_train = []
        self.y_train = []
        self.X_test = []

        self.y_test = []
        self.y_predict = []

        self.parser = iter( self.parse() )

    def parse( self ):
        g = gzip.open(self.file_name, 'r')

        for l in g:
            yield eval(l)
            # print type(l)

    def extract_features( self, sample_max = 100000 ):
        # df = pd.DataFrame()
        # data = pd.DataFrame({"A": range(3)})

        self.N_words = []
        self.N_unique_words = []
        self.y = []

        for cnt in xrange( sample_max ):

            review = self.parser.next()

            helpfulness = review['helpful']

            text = review['reviewText']
            tokenlizeText = text.split(' ')

            n_words = len(tokenlizeText)
            unique_words = [ word for word in tokenlizeText if word.strip(string.punctuation).lower() not in self.stop]
            n_unique_words = len(unique_words)

            if helpfulness[1] > 5:
                self.N_words.append( n_words )
                self.N_unique_words.append( n_unique_words )
                rating = float(helpfulness[0]) / helpfulness[1]
                self.y.append( rating )

            if cnt == 0:
                print text

            cnt += 1
            if cnt > sample_max:
                break

    def getData(self, data = "Training"):

        if data == "Training":
            self.X_train.append( self.N_words )
            self.X_train.append( self.N_unique_words )
            self.X_train = np.array( self.X_train ).T
            self.y_train = self.y

        elif data == "Testing":
            self.X_test.append( self.N_words )
            self.X_test.append( self.N_unique_words )
            self.X_test = np.array( self.X_test ).T
            self.y_test = self.y

        else:
            print "Warning: unknown data set."



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

    def showResult( self ):
        x = range(len(self.y_predict))
        plt.plot( x, self.y_test, 'r' )
        plt.plot( x, self.y_predict, 'k' )
        plt.legend(["True", "Prediction"])
        # plt.plot( x, (self.model.coef_[0] * +self.model.coef_[1]) * np.array(x) + self.model.intercept_, 'k' )
        plt.ylim((-0.5,1.5))
        plt.show()



if __name__ == "__main__":
    file_name = 'reviews_Books_5.json.gz'
    rq = calcReviewQuality( file_name )

    rq.extract_features( sample_max = 2000 )
    rq.getData(data = "Training")

    rq.extract_features( sample_max = 2000 )
    rq.getData( data = "Testing")

    rq.train_model( model = 'LinearRG') #'LogisticRG')#'LogisticRG' )
    rq.predict()

    print rq.y_test[:10], rq.y_predict[:10]
    print mse( rq.y_test, rq.y_predict )

    rq.showResult()




    # total number of words,
    # number of unique words,
    # number of paragraphs,
    # number of sentences

        # review['reviewerID']
        # review['reviewText'] #
        # review['overall']  # score
        # review['helpful']  # label






    # 'asin': '000100039X'
