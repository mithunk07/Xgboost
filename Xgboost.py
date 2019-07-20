#Import the Libraries
import pandas as pd
import numpy as np

#Import the dataset
heart=pd.read_csv('dataset.csv')

heart.isnull().sum()

#Data Preprocessing
heart = pd.get_dummies(heart, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
heart[columns_to_scale] = standardScaler.fit_transform(heart[columns_to_scale])


y = heart['target']
X = heart.drop(['target'], axis = 1)

#Split the dataset

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Xgboost Classifier model

# import xgboost library
import xgboost

xgb = xgboost.XGBClassifier(n_estimators=70, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb.fit(x_train,y_train)


#predict values
y_pred = xgb.predict(x_test)

#Evaluation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
report=classification_report(y_test,y_pred,target_names=target_names)
print(report)

