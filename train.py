from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor# big powerful for predicting the number 
from sklearn.model_selection import train_test_split#check data before training
from sklearn.metrics import mean_absolute_error,r2_score#check how far prediction is from actual price and how is model working
import pandas as pd #use panda for tabular data similar to excel spreadsheet
import joblib #to store train data and file in system hardware 

data = fetch_california_housing()

x=pd.DataFrame(data.data,columns=data.feature_names)
y=data.target #here prediction
# check data loaded or not
print(f"total recored : {x.shape}") #'tuple' object is not callable

# split the training data and training data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)\
# we need to test our model for the our data  to check how it perform on unseen data

# training a model
model=RandomForestRegressor(
    n_estimators=100,#100 means 100 decision tree internally combine and each tree predicting .avg prediction it give.  
    random_state=42 #more decision tree make training slow and more aq
)

model.fit(x_train,y_train)#model put on x.train and y.train and fit() method belonging to the model object.

y_pred=model.predict(x_test)
mae=mean_absolute_error(y_test,y_pred)
print(f"Avg Error: ${mae*100000:,.0f}")#:,.0f or :,.2f, depending on how many decimal places you want.

# Save the trained model
joblib.dump(model, "house_model_joblib")

# Save the feature columns
joblib.dump(list(x.columns), "house_features_joblib")
