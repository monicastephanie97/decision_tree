#Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 

#Importing Data
data_titanic=pd.read_csv("train.csv")

#Count Missing Values
data_titanic.isnull().sum()

#Remove Irrelevant Columns
for x in ["PassengerId","Cabin","Ticket"]:
    data_titanic=data_titanic.drop([x],axis=1)
    
#Filling Missing Quantitative Values with Mean
data_titanic["Age"]=data_titanic["Age"].fillna(data_titanic["Age"].mean())

#Filling Missing Qualitative Values with Mean
data_titanic["Embarked"]=data_titanic["Embarked"].fillna(data_titanic["Embarked"].mode()[0])

#Determining Dataset
dataset=data_titanic
for x in ["Name","Survived"]:
    dataset=dataset.drop([x], axis=1)
    
#Determining Data Result named as Data_Label
data_result=data_titanic["Survived"]

#Transforming Categorical Data into Numeric
categorical_label=["Pclass","Sex","Embarked"]

data_categorical=pd.DataFrame()
for x in categorical_label:
    data_dummy=pd.get_dummies(data_titanic[x],prefix=str(x))
    data_categorical=pd.concat([data_categorical,data_dummy],axis=1)
    
#Removing Categorical Columns
dataset=pd.concat([dataset,data_categorical],axis=1)
for x in categorical_label:
    dataset=dataset.drop([x],axis=1)
    
#Dividing Dataset into Data Train and Data Test
X_train, X_test, y_train, y_test=train_test_split(dataset,data_result,test_size=0.20,random_state=14)

#Use Decision Tree
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
data_predict=dt.predict(X_test)

#Compute Accuracy Score
print(accuracy_score(data_predict,y_test))

#Export The Decision Tree
from sklearn import tree
tree.export_graphviz(dt,out_file="GambarDecisionTree",
                    feature_names=list(dataset.columns),
                    class_names=['0','1'],
                    filled=True)