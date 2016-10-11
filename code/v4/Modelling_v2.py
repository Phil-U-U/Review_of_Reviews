import numpy as np
import pandas as pd

import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import graphlab as gl

from cntUppercases import cntUpperCaseLetters

from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error as mse

from scanCnt import scanWord
from getRawData import getRawData
from featureEngineering import feature_engineering


class calcReviewQuality( object ):

    def train_test_split( self, df, ratio = 0.4 ):

        target = df['Helpfullness']
        data = df.drop('Helpfullness', axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( data, target, test_size=ratio, random_state=1)


    def createHelpfulLevel( self ):

        helpful_ratio = self.y_train

        helpful_temp = np.empty( [ len(helpful_ratio), 1], dtype = object )

        # for i, val in enumerate(helpful_ratio):
        #     if val < 0.8:
        #         helpful_temp[i] = '0'
        #     else:
        #         helpful_temp[i] = '1'

        for i, val in enumerate(helpful_ratio):
            if val <= 0.5:
                helpful_temp[i] = '1'
            elif val <= 0.8 :
                helpful_temp[i] = '2'
            elif val <= 0.9:
                helpful_temp[i] = '3'
            else:
                helpful_temp[i] = '4'


        self.X_train['Helpful_category'] = helpful_temp

        helpfulness_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))),
                                    ('tfidf', TfidfTransformer()),
                                    ('clf', MultinomialNB())])


        helpfulness_clf.fit( self.X_train['corpus'], self.X_train['Helpful_category'] )


        self.X_test['Helpful_category'] = helpfulness_clf.predict( self.X_test['corpus'] )




# ******************** Summarize Features ********************

    def organizeFeatures( self ):

        features_category = ['Helpful_category', 'NUniquePuncs_category']#, 'NWords_category']
        X_train_category = self.X_train[ features_category ]
        X_train_category = pd.get_dummies( X_train_category, prefix = features_category )
        self.X_train = self.X_train.drop('corpus', axis=1)
        self.X_train = self.X_train.join( X_train_category )
        self.X_train = self.X_train.drop( features_category , axis = 1 )
        print self.X_train.head(2)


        X_test_category = self.X_test[ features_category ]
        X_test_category = pd.get_dummies( X_test_category, prefix = features_category )
        self.X_test = self.X_test.drop('corpus', axis=1)
        self.X_test = self.X_test.join( X_test_category )
        self.X_test = self.X_test.drop( features_category, axis = 1 )
        print self.X_train.head(2)



# ************************ Train model ************************
    def train_model( self, model = 'RandomForest' ):
        if model == 'LinearRG':
            self.model = LinearRegression()
        elif model == 'LogisticRG':
            self.model = LogisticRegression()
        elif model == 'RandomForest':

            self.model = RandomForestRegressor(n_jobs = 8, random_state = 1, max_features = "auto") #, min_samples_leaf = 50 )
        elif model == 'GradientBoostingRegressor':
            self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
            max_depth=1, random_state=0, loss='ls')


        if model == 'RandomForest':
            param_grid = { "n_estimators": [250, 300],
                        "max_depth": [10, 20]},
                        # "min_samples_split": [2, 4]}
                        # "boostrap": [True, False] }
            self.grid_search = GridSearchCV( self.model, param_grid = param_grid, n_jobs = 8, cv = 10 )
            self.grid_search.fit( self.X_train, self.y_train )


        else:
            self.model.fit( self.X_train, self.y_train )

    def predict( self, model = 'RandomForest' ):

        if model == 'RandomForest':
            self.y_predict = self.grid_search.predict( self.X_test )

        else:
            self.y_predict = self.model.predict( self.X_test )


    def showResult( self, len_show = 100 ): # len_show = len(self.y_predict)
        # df = pd.read_csv('Test_Data_10_10.csv')

        x = range(len(self.y_predict[:len_show]))
        plt.plot( x, self.y_test[:len_show], 'ro--' )
        plt.plot( x, self.y_predict[:len_show], 'ko--' )
        plt.legend(["True", "Prediction"])
        # plt.plot( x, (self.model.coef_[0] * +self.model.coef_[1]) * np.array(x) + self.model.intercept_, 'k' )
        plt.ylim((-0.5,1.5))
        plt.show()

    def graphLab( self ):
        # data = gl.SFrame(self.df)
        # data.show(view = "Summary")
        data = gl.SFrame.read_csv('Test_Data_10_11_v2.csv')
        data.show(view = "Summary")

    def save2DataFrame( self ):
        data_train = self.X_train.join(self.y_train)
        data_test  = self.X_test.join(self.y_test)
        data_test.join( pd.DataFrame(self.y_predict))

        data_train.to_csv( 'Training_Data_10_11_v2.csv' )
        data_test.to_csv( 'Test_Data_10_11_v2.csv' )



if __name__ == "__main__":
    file_name = 'reviews_Books_5.json.gz'

    df_rawData = getRawData( file_name ).read( sample_max = 100000, votes = 8 )

    fE = feature_engineering( df_rawData )
    fE.createNWordsLabel()
    fE.createNUniqueWordsLabel()
    fE.createNUniquePuncsLabel()

    crq = calcReviewQuality()
    crq.train_test_split( fE.df )
    crq.createHelpfulLevel()
    crq.organizeFeatures()

    crq.train_model()
    crq.predict()

    print crq.y_test[:10], crq.y_predict[:10]
    print mse( crq.y_test, crq.y_predict )

    crq.save2DataFrame()
    crq.showResult()
    # rq.graphLab()
