from sklearn.preprocessing import scale
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
import time
def sequentialfeatureselector(df):
    st = time.time()
    X_train = df.iloc[:, :-1].values   
    y_train = df.iloc[:, 9].values
    X_train_scaled = scale(X_train)
    sfs = SequentialFeatureSelector(LinearRegression(), forward=False, k_features=9)
    sfs.fit(X_train_scaled, y_train)
    et = time.time()
    print(sfs.k_feature_idx_)
    column_headers = list(df.columns)
    print('\nlist of best feature in sequentialfeatureselector methods:\n')
    for i in range(9):
        if sfs.k_feature_idx_[i]>=5:
            print(i,column_headers[i])
    elapsed_time = et - st
    print()
    print('sequentialfeatureselector running time:', elapsed_time, 'seconds',)
