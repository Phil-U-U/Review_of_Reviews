from flask import Flask, request, render_template
import pandas as pd
from scanText import scanText
import cPickle as pickle

# from featureEngineering import feature_engineering

# Initialize your app and load your pickled models.
# =================================================
# init flask app
app = Flask(__name__)

# load the model pickled in build_model.py
with open('data/model.pkl') as f:
    model = pickle.load(f)

with open('data/helpfulness_clf.pkl') as f:
    helpfulness_clf = pickle.load(f)

# Homepage with welcome informatoion on it.
# =================================================
@app.route('/')
def HomePage():
    return render_template('homepage.html')
    # return '<h1> Hi All! Welcome to Review Evaluation System!</h1>'

# # Create page with a form for submit reviews
# @app.route('/submit')
# def submit():
#     return '''
#     <form action = "/predict" method = 'POST'>
#         <input type="text" name="user_input" />
#         <input type="submit" />
#     </form>
#     '''

# Once submit is hit, pass info into model, return results.
# =================================================
@app.route('/predict', methods = ['POST'])
def predict_helpfulness():
    # get data from request form, the key is the name you set in your form
    book_name = str( request.form['BookName'] )

    # convert data from unicode to string
    review_text = str( request.form['ReviewText'] )

    # abstract features from data
    # features = abstract_features( data )
    n_sentences, avg_wordsPerSent, n_words, n_unique_words, n_upperCases, n_puncs, n_puncs_unique, isBrackets = scanText().scan(review_text)


    df_rawData = pd.DataFrame({
                            "N_sentences": n_sentences,
                            "Avg_wordsPerSent": avg_wordsPerSent,
                            "IsBrackets": isBrackets,
                            "N_words": n_words,
                            "N_unique_words": n_unique_words,
                            "N_upperCases": n_upperCases,
                            "N_puncs": n_puncs,
                            "N_puncs_unique": n_puncs_unique
                            }, index=[0])

    # fE = feature_engineering( df_rawData )
    # fE.createNWordsLabel()
    # fE.createNUniqueWordsLabel()
    # fE.createNUniquePuncsLabel()

    df_rawData['Helpful_category'] = helpfulness_clf.predict( [review_text] )

    features_category = ['Helpful_category',  'IsBrackets']  # 'NUniquePuncs_category',
    X_test_category = df_rawData[ features_category ] # features_category = ['Helpful_category', 'NUniquePuncs_category', 'IsBrackets']


    complement_values = pd.DataFrame( 1-X_test_category.values.astype(int), columns = features_category )
    X_test_category = X_test_category.append( complement_values, ignore_index = True )

    X_test_category = pd.get_dummies( X_test_category, prefix = features_category )
    # print X_test_category
    df_rawData = df_rawData.join( X_test_category )
    df_rawData = df_rawData.drop( features_category, axis = 1 )
    features = df_rawData[0:1]

    # make prediction based on new data
    pred = model.predict(features)

    # return a string format of the prediction to the html page
    return render_template('prediction_template.html', prediction=str(pred), reviewText = review_text, bookName = book_name )






if __name__ == '__main__':
    app.run( host='0.0.0.0', port=8002, debug=True )
