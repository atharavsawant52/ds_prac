#Practical 7 – Logistic Regression (Ultra Short)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X,y = load_iris(return_X_y=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)

print("Accuracy:",accuracy_score(y_test,model.predict(X_test)))


#Decision Tree

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

print("Accuracy:",accuracy_score(y_test,model.predict(X_test)))