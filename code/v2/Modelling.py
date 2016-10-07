import numpy as np
import pandas as pd


import gzip

import string
#from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error as mse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt



class calcReviewQuality( object ):

    def __init__( self, file_name ):
        self.file_name = file_name
        self.stop = set(stopwords.words('english'))
        self.wnl = WordNetLemmatizer()

        self.model = None
        self.X_train = pd.DataFrame()
        self.y_train = []
        self.X_test = pd.DataFrame()
        self.y_test = []
        self.y_predict = []
        self.Helpful_category = None

        self.parser = iter( self.parse() )

    def parse( self ):
        g = gzip.open(self.file_name, 'r')

        for l in g:
            yield eval(l)

# ************************* Acquire Raw Data *************************
    def getRawData( self, sample_max = 100000 ):

        self.corpus = []
        self.N_words = []
        self.N_unique_words = []
        self.helpful_ratio = []
        self.helpful_pair = []
        self.cntNoVotes = 0
        self.cntVotes_less_than_5 = 0
        self.summary = []
        self.df = None

        for cnt in xrange( sample_max ):

            review = self.parser.next()

            helpfulness = review['helpful']

            if helpfulness[1] >= 5:
                text = review['reviewText']
                tokenlizeText = text.split(' ')

                n_words = len(tokenlizeText)
                unique_words = [ word for word in tokenlizeText if word.strip(string.punctuation).lower() not in self.stop]
                n_unique_words = len(unique_words)

                self.corpus.append(text)
                self.N_words.append( n_words )
                self.N_unique_words.append( n_unique_words )
                rating = float(helpfulness[0]) / helpfulness[1]
                self.helpful_ratio.append( rating )
                self.helpful_pair.append( (helpfulness[0], helpfulness[1]) )
                self.summary.append( (n_words, n_unique_words, (helpfulness[0], helpfulness[1]), rating) )
            else:
                self.cntVotes_less_than_5 += 1

            if cnt == sample_max-1:
                # print "Number of words:{}".format(self.N_words)
                # print "Number of unique words:{}".format(self.N_unique_words)
                # print "Helpful pair:{}".format(self.helpful_pair)
                # print "Helpful ratio:{}".format( self.helpful_ratio )
                # print "Number of 0 votes:{}".format( self.cntNoVotes )
                # print "Number of votes less than 5:{}".format( self.cntVotes_less_than_5 )
                # print "Number of 0 votes:{}".format( self.cntNoVotes )
                # print "Summary:{}".format( self.summary )

                self.df = pd.DataFrame({"Helpfullness": self.helpful_ratio,
                                        "N_words": self.N_words,
                                        "N_unique_words": self.N_unique_words,
                                        "helpful_pair": self.helpful_pair
                                        })
                break

        print "Votes ratio( >5 vs all): {}".format(1-self.cntVotes_less_than_5/float(sample_max))

# ************************* Feature Engineering *************************
    def createNWordsLabel( self ):
        NWords_category = np.empty( [ len(self.N_words), 1], dtype = object )
        # print self.N_words
        for i, val in enumerate(self.N_words):
            if val <= 100:
                NWords_category[i] = '1'
            elif val <= 200:
                NWords_category[i] = '2'
            elif val <= 350:
                NWords_category[i] = '3'
            else:
                NWords_category[i] = '4'
        self.df['NWords_category'] = NWords_category

    def createNUniqueWordsLabel( self ):
        NUniqueWords_category = np.empty( [ len(self.N_unique_words), 1], dtype = object )
        for i, val in enumerate(self.N_unique_words):
            if val <= 50:
                NUniqueWords_category[i] = '1'
            elif val <= 100:
                NUniqueWords_category[i] = '2'
            elif val <= 200:
                NUniqueWords_category[i] = '3'
            else:
                NUniqueWords_category[i] = '4'
        self.df['NUniqueWords_category'] = NUniqueWords_category

# ***** key features ******
    def createHelpfulLevel_Training( self ):
        helpful_category = np.empty( [ len(self.helpful_ratio), 1], dtype = object )
        for i, val in enumerate(self.helpful_ratio):
            if val <= 0.5:     # 0 ~ 0.5
                helpful_category[i] = '1'
            elif val <= 0.9:   # 0.5 ~ 0.8
                helpful_category[i] = '2'
            # elif val <= 0.9:              # 0.8 ~ 1
            #     helpful_category[i] = '3'
            elif val <= 1:
                helpful_category[i] = '3'
            else:
                print "Wrong Helpful category!"

        helpfulness_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB())])
        self.df['Helpful_category'] = helpful_category
        self.helpfulness_clf = helpfulness_clf.fit( self.corpus, helpful_category )


    def createHelpfulLevel_Testing( self ):
        self.helpful_category = self.helpfulness_clf.predict( self.corpus )
        print self.helpful_category


        self.df['Helpful_category'] = self.helpful_category



# ******************** Summarize Features ********************

    def organizeFeatures(self, data = "Training"):

        if data == "Training":
            # self.X_train.append( self.df['NWords_category'] )
            # self.X_train.append( self.df['NUniqueWords_category'] )
            # self.X_train.append( self.df['Helpful_category'] )
            self.X_train = self.df[['NWords_category', 'NUniqueWords_category', 'Helpful_category']]

            # self.X_train = np.array( self.X_train ).T
            self.X_train = pd.get_dummies( self.X_train, prefix = ['NWords_category', 'NUniqueWords_category', 'Helpful_category'] )

            self.y_train = self.df['Helpfullness']

        elif data == "Testing":
            # self.X_test.append( self.df['NWords_category'] )
            # self.X_test.append( self.df['NUniqueWords_category'] )
            # self.X_test.append( self.df['Helpful_category'] )
            self.X_test = self.df[['NWords_category', 'NUniqueWords_category', 'Helpful_category']]

            # self.X_test = np.array( self.X_test ).T
            self.X_test = pd.get_dummies( self.X_test, prefix = ['NWords_category', 'NUniqueWords_category', 'Helpful_category'] )

            self.y_test = self.df['Helpfullness']

            print self.X_train.head()
            print self.X_test.head()

        else:
            print "Warning: unknown data set."

# ************************ Train model ************************
    def train_model( self, model = 'GradientBoostingRegressor' ):
        if model == 'LinearRG':
            self.model = LinearRegression()
        elif model == 'LogisticRG':
            self.model = LogisticRegression()
        elif model == 'RandomForest':
            self.model = RandomForestRegressor(n_estimators=10)
        elif model == 'GradientBoostingRegressor':
            self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
            max_depth=1, random_state=0, loss='ls')


        self.model.fit( self.X_train, self.y_train )
        # The coefficients
        # print('Coefficients: \n', self.model.coef_)

    def predict( self ):
        self.y_predict = self.model.predict( self.X_test )

    def showResult( self ):
        x = range(len(self.y_predict[:100]))
        plt.plot( x, self.y_test[:100], 'ro--' )
        plt.plot( x, self.y_predict[:100], 'ko--' )
        plt.legend(["True", "Prediction"])
        # plt.plot( x, (self.model.coef_[0] * +self.model.coef_[1]) * np.array(x) + self.model.intercept_, 'k' )
        plt.ylim((-0.5,1.5))
        plt.show()



if __name__ == "__main__":
    file_name = 'reviews_Books_5.json.gz'
    rq = calcReviewQuality( file_name )

    rq.getRawData( sample_max = 50000 )
    rq.createNWordsLabel()
    rq.createNUniqueWordsLabel()
    rq.createHelpfulLevel_Training()
    rq.organizeFeatures(data = "Training")

    rq.getRawData( sample_max = 50000 )
    rq.createNWordsLabel()
    rq.createNUniqueWordsLabel()
    rq.createHelpfulLevel_Testing()
    rq.organizeFeatures(data = "Testing")

    rq.train_model() #'LogisticRG')#'LogisticRG' )
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
