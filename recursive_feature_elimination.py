from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import scale
import time
## create ranking among all features by selecting only one
def ref(df):
    st = time.time()
    X_train = df.iloc[:, :-1].values   
    y_train = df.iloc[:, 9].values
    X_train_scaled = scale(X_train)
    rfe = RFE(LinearRegression(), n_features_to_select=1)
    rfe.fit(X_train_scaled, y_train)
    et = time.time()
    print(rfe.ranking_)
    column_headers = list(df.columns)
    print('\nlist of best feature in Recursive Feature Eliminate methods:\n')
    for i in range(9):
        if rfe.ranking_[i]>=5:
            print(i,column_headers[i])
    elapsed_time = et - st
    print()
    print('ref running time:', elapsed_time, 'seconds',)
