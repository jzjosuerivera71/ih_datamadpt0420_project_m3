# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import cross_val_score


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#       print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# DATA
diamonds  = pd.read_csv('../input/dataptmad0420/diamonds_train.csv')
diamonds_predict = pd.read_csv('../input/dataptmad0420/diamonds_predict.csv')
sample_sub = pd.read_csv('../input/dataptmad0420/sample_submission.csv')

#VARIABLES
NUM_FEATS = ['carat', 'x', 'y', 'z']
CAT_FEATS = ['cut', 'color', 'clarity']
FEATS = NUM_FEATS + CAT_FEATS
TARGET = 'price'

#TRANSFORMERS
numeric_transformer = \
Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), 
                ('scaler', StandardScaler())])


categorical_transformer = \
Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = \
ColumnTransformer(transformers=[('num', numeric_transformer, NUM_FEATS),
                                ('cat', categorical_transformer, CAT_FEATS)])


# SPLIT
#diamonds_train, diamonds_test = train_test_split(diamonds)


#MODEL

model= Pipeline(steps=[('preprocessor', preprocessor),
                       ('regressor', xgb.XGBRegressor(objective="reg:linear", random_state=42,
                                                      colsample_bytree= 0.9315645941794735,
                                                     gamma= 0.006101536159022258,
                                                     learning_rate= 0.2095328103615681,
                                                     max_depth=5,
                                                     n_estimators= 135,
                                                     subsample= 0.8864716298931808))])

model.fit(diamonds[FEATS],diamonds[TARGET])


#Prediction

#y_test = model.predict(diamonds_test[FEATS])
#y_train = model.predict(diamonds_train[FEATS])

y_pred = model.predict(diamonds_predict[FEATS])

#Submision

submission_df_2 = pd.DataFrame({'id': diamonds_predict['id'], 'price': y_pred})

submission_df_2.price.clip(0, 20000, inplace=True)

submission_df_2.to_csv('diamonds_rf_2.csv', index=False)
