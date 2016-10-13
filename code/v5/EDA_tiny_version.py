import numpy as np
import pandas as pd
import gzip

import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error as mse

import graphlab as gl
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class EDA( object ):

    def __init__(self):
        self.parser = iter( self.parse() )
        self.stop = set(stopwords.words('english'))

    def parse( self ):
        file_name = 'reviews_Books_5.json.gz'
        g = gzip.open(file_name, 'r')

        for l in g:
            yield eval(l)

    def extract_features( self, n_sample = 100000 ):

        self.corpus = []
        self.N_words = []
        self.N_unique_words = []
        self.helpful_ratio = []
        self.helpful_pair = []
        self.cntNoVotes = 0
        self.cntVotes_less_than_5 = 0
        self.summary = []
        self.df = None



        for cnt in xrange( n_sample ):

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

            if cnt == n_sample-1:
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
                print self.df.describe()

                # self.df.boxplot()
                # scatter_matrix( self.df, figsize = (10,10) )#, dagonal = 'kde' )
                # plt.show()
                break

        print "Votes ratio(<5 vs all): {}".format(self.cntVotes_less_than_5/float(n_sample))

    def show_features( self, xlabel = 'N_words' ):

        if xlabel == 'N_words':
            feature = self.N_words
        elif xlabel == 'N_unique_words':
            feature = self.N_unique_words
        else:
            pass

        n_bins = 10
        max_N_unique = max(feature)

        bins = np.linspace(0, max_N_unique, n_bins, True).astype(np.int)
        sum_helpfulness = np.array( [0.0]*len(bins) )
        cnt_votes = np.array( [1]*len(bins) )
        # average_helpfulness = np.array( [0]*len(bins) )

        for i in xrange( len(feature) ):
            idx_bin = float(feature[i]) / bins[1]
            # print idx_bin
            sum_helpfulness[idx_bin] += self.helpful_ratio[i]
            cnt_votes[idx_bin] += 1

        print bins
        # print sum_helpfulness
        print cnt_votes
        average_helpfulness = sum_helpfulness/cnt_votes

        # for i in xrange( len(sum_helpfulness) ):
        #     average_helpfulness[i] = float(sum_helpfulness[i]) / (cnt_votes[i]+1)

        # df = pd.DataFrame({'slices':slices, 'Average_Helpfulness':yy})
        # scatter_matrix( df, figsize = (10,10) )#, dagonal = 'kde' )

        # print df
        # print yy
        print average_helpfulness

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Effect of ' + xlabel + ' on Helpfulness' )
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Average Helpfulness')
        ax.plot( bins, average_helpfulness, 'ro--' )
        plt.xticks( bins )
        plt.show()

    def graphLab( self ):
        data = gl.SFrame(self.df)
        data.show(view = "Summary")

    def tf_idf( self ):
        # tf-idf
        # from sklearn.feature_extraction.text import CountVectorizer
        # from sklearn.feature_extraction.text import TfidfTransformer
        # vectorizer = CountVectorizer(min_df=1)
        # # bigram_vectorizer = CountVectorizer( ngram_range(1, 2),
        # # min_df = 1 )
        #
        # counts_1 = vectorizer.fit_transform(self.corpus)
        # # counts_2 = bigram_vectorizer.fit_transform( self.corpus )
        #
        # transformer = TfidfTransformer( smooth_idf = True )
        # transformer.fit_transform( counts_1 )
        # self.tf_idf = transformer.idf_

        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer( min_df = 1 )
        vectorizer.fit_transform( self.corpus )
        self.tf_idf_res = vectorizer.transform(['Test this statement to see if it works.'])

        # self.df['tf_idf'] = vectorizer.idf_
        # vectorizer.transform(['Something completely new.'])

        # from sklearn.feature_extraction.text import HashingVectorizer
        # hv = HashingVectorizer( n_features = 5 )
        # hv.transform(self.corpus)
        # print len(hv.idf_)

    def createHelpfulLabel( self ):
        self.helpful_category = np.empty( [ len(self.helpful_ratio), 1], dtype = object )
        for i, val in self.helpful_ratio:
            if val <= 0.5:     # 0 ~ 0.5
                self.helpful_category[i] = 1
            elif val <= 0.8:   # 0.5 ~ 0.8
                self.helpful_category[i] = 2
            # elif val <= 0.:    # 0.5 ~ 0.8
            #     self.helpful_category[i] = 3
            else:              # 0.8 ~ 1
                self.helpful_category[i] = 4

        self.df['Helpful_category'] = self.helpful_category



    def helpful_clf( self ):
        helpfulness_clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB())])

        self.helpfulness_clf = helpfulness_clf.fit(self.corpus, self.helpful_category)

    def createHelpfulLevel( self ):
        self.helpfulness_clf.predict( self.test )

    def save2DataFrame( self ):
        self.df.to_csv( 'Features.csv' )






if __name__ == '__main__':

    eda = EDA()
    eda.extract_features()
    # eda.show_features()
    eda.show_features(xlabel = 'N_unique_words')
    # eda.graphLab()
    # eda.tf_idf()
    eda.save2DataFrame()
