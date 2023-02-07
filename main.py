
'''                      Feature Selection Methods                  '''

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<liberaries:

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from univariant_method import*
from model_based_method import*
from changing_lasso_method import*
from recursive_feature_elimination import*
from sequentialfeatureselector_method import*
from Random_Forest import*
from sklearn.utils import shuffle
import seaborn as sns
from unsupervised_pca import*
#----------------------------------------
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Load cancer dataset:

df = pd.read_csv('Bc.csv')
print(df)
#----------------------------------------
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Statistics analysis:

print('\n dataset description\n')
print(df.describe())
print(df.info())
print(df.isnull().sum())
column_headers = list(df.columns)
feature_len = len(column_headers)-1
print('\n')
print('\n Features name:\n')
print(column_headers)
#matrix covariance:
correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
print('\n dataset description\n')
print(df.describe())

plt.show()
fig = plt.figure(figsize =(5, 5))
# Creating plot

plt.boxplot(df)
hist = df.hist(bins=3)
# show plot
plt.show()
#----------------------------------------
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Datset normalization:

# copy the data
df_max_scaled = df.copy()
  
# apply normalization techniques
for column in df_max_scaled.columns:
    df_max_scaled[column] = df_max_scaled[column]/df_max_scaled[column].abs().max()
      
df_max_scaled.plot(kind = 'bar')
df = df_max_scaled
#----------------------------------------
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Dataset seperation to X and Y

X = df.iloc[:, :-1].values   
Y = df.iloc[:, 9].values
#----------------------------------------
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Feature selection methods:
print()
print()
print('\n *****************Univarint method resuls*****************\n')
univariant(df)
print()
print('\n *****************Model based (single fit) methods resuls*****************\n')
model_based(df)
print()
print('\n *****************changing_Lasso_alpha methods resuls*****************\n')
changing_Lasso_alpha(df)
print()
print('\n *****************recursive_feature_elimination methods resuls*****************\n')
ref(df)
print()
print('\n *****************sequentialfeatureselector methods resuls*****************\n')
print(sequentialfeatureselector(df))
print()
print('\n *****************PCA Unsupervised methods resuls*****************\n')
pca(df)
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<TEST Feature selection by Random Forest
print()
print()
df = shuffle(df)
comparison = []
print('\n *****************Original data set prediction*****************\n')
X = df.iloc[:, :-1].values   
y = df.iloc[:, 9].values

print('\n *****************Univarint Test prediction*****************\n')
X = df[[ 'Cl.thickness','Cell.size','Cell.shape','Bl.cromatin','Normal.nucleoli']]  

comparison.append(RF(X,y))
print('\n *****************Model based (single fit) Test prediction*****************\n')
X = df[[ 'Cl.thickness','Cell.size','Cell.shape','Bl.cromatin','Marg.adhesion']]  

comparison.append(RF(X,y))
print('\n *****************changing_Lasso_alpha methods Test prediction*****************\n')
X = df[['Id','Cl.thickness','Cell.size','Cell.shape','Bl.cromatin','Marg.adhesion','Normal.nucleoli','Epith.c.size']]  

comparison.append(RF(X,y))
print('\n *****************recursive_feature_elimination methods Test prediction*****************\n')
X = df[['Id','Marg.adhesion','Normal.nucleoli','Epith.c.size','Mitoses']]  

comparison.append(RF(X,y))
print('\n *****************sequential featureselector  methods Test prediction*****************\n')
X = df[['Bl.cromatin','Normal.nucleoli','Epith.c.size','Mitoses']]  

comparison.append(RF(X,y))
print('\n ***************** PCA  methods Test prediction*****************\n')
X = df[['Cl.thickness','Cell.size','Epith.c.size','Marg.adhesion','Cell.shape']]  

comparison.append(RF(X,y))
plt.show()
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<SHOW the pichart comparison>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
c = ['Univarint','Model based','changing_Lasso_alpha ','recursive_feature_elimination ','sequential featureselector','PCA']
y =np.array(comparison)
plt.pie(y, labels = c)
plt.show()
