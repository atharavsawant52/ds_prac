#Simple Linear Regression

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
data = {
    "Rooms":[2,3,4,5,6],
    "Price":[100,150,200,250,300]
}

df = pd.DataFrame(data)

X = df[["Rooms"]]
y = df["Price"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("MSE:",mean_squared_error(y_test,y_pred))
print("R2 Score:",r2_score(y_test,y_pred))
print("Intercept:",model.intercept_)
print("Coefficient:",model.coef_)


#Multiple Linear Regression

import pandas as pd

from sklearn.linear_model import LinearRegression

data={
    "Rooms":[2,3,4,5,6],
    "Area":[800,1000,1200,1500,1800],
    "Price":[100,150,200,250,300]
}

df=pd.DataFrame(data)

X=df[["Rooms","Area"]]

y=df["Price"]

model=LinearRegression(
)

model.fit(X,y)

print(model.intercept_)
print(model.coef_)