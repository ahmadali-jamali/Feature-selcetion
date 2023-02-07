from sklearn.linear_model import LassoCV
from sklearn.preprocessing import scale
import time
def model_based(df):
    st = time.time()
    X_train = df.iloc[:, :-1].values   
    y_train = df.iloc[:, 9].values
    
    X_train_scaled = scale(X_train)
    lasso = LassoCV().fit(X_train_scaled, y_train)
    et = time.time()
    print(lasso.coef_)
    column_headers = list(df.columns)
    print('\nlist of best feature in model_based methods:\n')
    for i in range(9):
        if lasso.coef_[i]>=0.05:
            print(i,column_headers[i])
    elapsed_time = et - st        
    print()
    print('model_based running time:', elapsed_time, 'seconds',)
