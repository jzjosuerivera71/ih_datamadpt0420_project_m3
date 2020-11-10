# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import cross_val_score

# DATA
diamonds  = pd.read_csv('../input/dataptmad0420/diamonds_train.csv')
diamonds_predict = pd.read_csv('../input/dataptmad0420/diamonds_predict.csv')
sample_sub = pd.read_csv('../input/dataptmad0420/sample_submission.csv')

diamonds['Vol'] = (0.2/3.51)*diamonds['carat']
diamonds_predict['Vol'] = (0.2/3.51)*diamonds_predict['carat']

NUM_FEATS = ['carat','depth', 'table','x', 'y', 'z','Vol'] #'Vol'
CAT_FEATS = ['cut', 'color', 'clarity']
FEATS = NUM_FEATS + CAT_FEATS
TARGET = 'price'

numeric_transformer = \
Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), #mean
                ('scaler', StandardScaler())])


categorical_transformer = \
Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = \
ColumnTransformer(transformers=[('num', numeric_transformer, NUM_FEATS),
                                ('cat', categorical_transformer, CAT_FEATS)])

model= Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', GradientBoostingRegressor(n_estimators=700,
                                                                min_samples_split=122,
                                                                max_depth= 9,
                                                                min_samples_leaf=10,
                                                                max_features=15,
                                                                subsample=0.9))])

model.fit(diamonds[FEATS],diamonds[TARGET])




#Prediction

#y_test = model.predict(diamonds_test[FEATS])
#y_train = model.predict(diamonds_train[FEATS])

y_pred = model.predict(diamonds_predict[FEATS])

#Submision

submission_df_4 = pd.DataFrame({'id': diamonds_predict['id'], 'price': y_pred})

submission_df_4.price.clip(0, 20000, inplace=True)

submission_df_4.to_csv('diamonds_rf_4.csv', index=False)
