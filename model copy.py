import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

Flower_data =pd.read_csv('iris.csv')
print(Flower_data.head())
x= Flower_data[['SL','SW','PL','PW']]
y = Flower_data[['Classification']]
Flower_data.drop(columns='Classification',inplace=True)
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=42, test_size=0.2)

from sklearn.linear_model import LogisticRegression

regressor= LogisticRegression()
regressor = regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

pickle.dump(regressor,open('model.pkl','wb'))