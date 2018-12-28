
# coding: utf-8

# In[2]:


# < > No tengo estos simbolos en el teclado por lo que los tengo aquí para después....
import numpy as np
import pandas as pd # Manipular los datos de forma sencilla para poder estudiarlos previamente

import matplotlib.pyplot as plt # Representaciones
import seaborn as sns # Representaciones de correlaciones entre variables
get_ipython().run_line_magic('matplotlib', 'inline')

import scipy.stats as stats # Predicciones estadísticas


# # 1 - Descarga y carga de datos
# Despues de descargar los archivos en la carpeta usual de descargas haremos el proceso de carga y primer estudio utilizando pandas.

# In[18]:


# Cargamos los datos en un DataFrame
df = pd.read_csv('C:/Users/Usuario/Downloads/train.csv/train.csv', nrows=100000)
df.head(5) #Mostramos un ejemplo de los datos (podría ser "df.sample()")
indices = df.MachineIdentifier


# In[32]:


df = df.select_dtypes('number')


# In[33]:


# drop NA columns
temp = df.isna().sum() / df.shape[0] > 0.1
bad_columns = temp[temp==True].index
bad_columns

df = df.drop(bad_columns, axis=1)


# In[34]:


df = df.fillna(method='ffill')


# In[35]:


from sklearn.model_selection import train_test_split


# In[9]:


X = df.drop('HasDetections', axis=1)
y = df['HasDetections']

# Separamos los datos en el conjunto de entrenamiento en el test
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[10]:


from sklearn.metrics import roc_auc_score


# In[11]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


# In[12]:


gbcv = GridSearchCV(GradientBoostingClassifier(), 
                   param_grid = {
                       'learning_rate':[0.1],
                       'n_estimators':[200],
                       'max_depth':[5]
                   }, verbose=2)
gbcv.fit(X_train, y_train)


# In[16]:


gbcv.best_estimator_


# In[13]:


train_score = roc_auc_score(y_train, gbcv.predict_proba(X_train)[:, 1])
test_score = roc_auc_score(y_test, gbcv.predict_proba(X_test)[:, 1])        


# In[14]:


train_score


# In[15]:


test_score


# ### Predict test

# In[39]:


cols = df.drop('HasDetections', axis=1).columns


# In[91]:


import os


# In[99]:


submission_path = 'C:/Users/Usuario/Downloads/'
submission_file_name = 'submission2.csv'


# In[101]:


if submission_file_name in os.listdir(submission_path):
    print('ya existe ese archivo, cuidado')
else:
    print('continua')


# In[106]:


submission_file_path = submission_path + submission_file_name


# In[107]:


with open(submission_file_path, 'w+') as file:
    file.write('MachineIdentifier,HasDetections\n')


# In[120]:


nrows_test = 7853253
chunksize = 10000


# In[121]:


for chunk_number, df_test in enumerate(pd.read_csv('C:/Users/Usuario/Downloads/test.csv', chunksize=chunksize)):
    print(f'{chunk_number * chunksize / nrows_test * 100: .1f}%')
    ids = df_test.MachineIdentifier
    df_test = df_test[cols]
    df_test = df_test.fillna(df_test.mean())
    
    preds = pd.DataFrame({'MachineIdentifier': ids, 'HasDetections': gbcv.predict_proba(df_test)[:, 1]})
    
    
    with open(submission_file_path, 'a') as file:
        preds.to_csv(file, index=False, header=False)

