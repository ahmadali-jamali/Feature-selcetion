from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import numpy 
import time

def univariant(df):
    st = time.time()
    X = df.iloc[:, :-1].values   
    Y = df.iloc[:, 9].values
    # feature extraction
    test = SelectKBest(score_func=f_classif, k=4)
    fit = test.fit(X, Y)
    # summarize scores
    set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(X)
    et = time.time()
    # summarize selected features
    print(features[0:3,:])
    s = numpy.sort(fit.scores_)
    
    print(s)
    column_headers = list(df.columns)
    print('\nlist of best feature in univariant methods:\n')
    for i in range(9):
        if fit.scores_[i]>=700:
            print(i,column_headers[i])
    elapsed_time = et - st
    print()
    print('univariant running time:', elapsed_time, 'seconds',)
