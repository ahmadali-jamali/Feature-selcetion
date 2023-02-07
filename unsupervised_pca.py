
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
import prince
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time
def pca(df):
    X_train = df.iloc[:, :-1].values 
    st = time.time()
    pca = prince.PCA(
         n_components=6,
         n_iter=10,
         rescale_with_mean=False,
         rescale_with_std=False,
         copy=True,
         check_input=True,
         engine='sklearn',
         random_state=234
     )
    pca = pca.fit(X_train)

    #print(pca.eigenvalues_)
    column_headers = list(df.columns)
    dset = pd.DataFrame()
    dset['pca'] = range(1,7)
    dset['eigenvalue'] = pd.DataFrame(pca.eigenvalues_)
    et = time.time()
    print('pca.eigenvalues:')
    print(pca.eigenvalues_)
    print(pca.explained_inertia_)
    print('\nlist of best feature in model_based methods:\n')
    for i in range(len(pca.explained_inertia_)):
        #if pca.explained_inertia_[i]>=0.021:
            print(i,column_headers[i])
    elapsed_time = et - st
    print()
    print('PCA_unsupervised running time:', elapsed_time, 'seconds',)
