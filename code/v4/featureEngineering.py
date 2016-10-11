import numpy as np
# ************************* Feature Engineering *************************
class feature_engineering( object ):
    def __init__( self, df ):
        self.df = df

    def createNWordsLabel( self ):
        NWords_category = np.empty( [ len(self.df['N_words']), 1], dtype = object )
        # print self.N_words
        for i, val in enumerate(self.df['N_words']):
            if val <= 100:
                NWords_category[i] = '1'
            elif val <= 300:
                NWords_category[i] = '2'
            elif val <= 600:
                NWords_category[i] = '3'
            else:
                NWords_category[i] = '4'
        self.df['NWords_category'] = NWords_category

    def createNUniqueWordsLabel( self ):
        NUniqueWords_category = np.empty( [ len(self.df['N_unique_words']), 1], dtype = object )
        for i, val in enumerate(self.df['N_unique_words']):
            if val <= 90:
                NUniqueWords_category[i] = '1'
            elif val <= 300:
                NUniqueWords_category[i] = '2'
            # elif val <= 200:
                # NUniqueWords_category[i] = '3'
            else:
                NUniqueWords_category[i] = '3'
        self.df['NUniqueWords_category'] = NUniqueWords_category

    def createNUniquePuncsLabel( self ):
        NUniquePuncs_category = np.empty( [ len(self.df['N_puncs_unique']), 1], dtype = object )
        for i, val in enumerate( self.df['N_puncs_unique'] ):
            if val <= 2:
                NUniquePuncs_category[i] = '1'
            elif val <= 8:
                NUniquePuncs_category[i] = '2'
            else:
                NUniquePuncs_category[i] = '3'
        self.df['NUniquePuncs_category'] =  NUniquePuncs_category
