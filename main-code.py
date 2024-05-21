
"Import Libaries "
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics


print("==================================================")
print("Qunatum Computing  Dataset")
print(" Process - QUANTUM MACHINE LEARNING OF BIG DATASETS USING RANDOMIZED MEASUREMENTS")
print("==================================================")
print("1.Data Selection ")
print("==================================================")

##1.data slection---------------------------------------------------
#def main():
dataframe=pd.read_csv("Quantumdataset.csv")
# dataframe=dataframe.iloc[:1000] 

# dataframe=dataframe.iloc[:100]
print("---------------------------------------------")
print()
print("Data Selection")
print("Samples of our input data")
print(dataframe.head(10))
print("----------------------------------------------")
print()


 #2.pre processing--------------------------------------------------
#checking  missing values 
print("2.Data Pre processing  ")
print("==================================================")
print("---------------------------------------------")
print()
print("Before Handling Missing Values")
print()
print(dataframe.isnull().sum())
print("----------------------------------------------")
print() 
    
print("-----------------------------------------------")
print("After handling missing values")
print()
dataframe_2=dataframe.fillna(0)
print(dataframe_2.isnull().sum())
print()
print("-----------------------------------------------")
 

#label encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
print("--------------------------------------------------")
print("Before Label Handling ")
print()
print(dataframe_2.head(10))
print("--------------------------------------------------")
print()


df_train_y=dataframe_2["label"]
df_train_X=dataframe_2.iloc[:,:20]
from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()

df_train_X['proto'] = number.fit_transform(df_train_X['proto'].astype(str))
df_train_X['service'] = number.fit_transform(df_train_X['service'].astype(str))
df_train_X['state'] = number.fit_transform(df_train_X['state'].astype(str))
#df_train_X['attack_cat'] = number.fit_transform(df_train_X['attack_cat'].astype(str))
print("==================================================")
print(" Preprocessing")
print("==================================================")

df_train_X.head(5)
x=df_train_X
y=df_train_y
    
print("3.Data feature Selection  ")
print("==================================================")
##4.feature selection------------------------------------------------
##kmeans
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x, y_true = make_blobs(n_samples=175341, centers=2,cluster_std=0.30, random_state=0)
plt.scatter(x[:, 0], x[:, 1], s=20);

kmeans = KMeans(n_clusters=2)
kmeans.fit(x)
y_kmeans = kmeans.predict(x)

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=20, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.title("Self-Organizing Map ")
plt.show()

#---------------------------------------------------------------------------------------
"4.Data Splitting "
print("4.Data Splitting  ")
print("==================================================")
x_train,x_test,y_train,y_test = train_test_split(df_train_X,y,test_size = 0.20,random_state = 42)
print("X_train Shapes ",x_train.shape)
print("y_train Shapes ",y_train.shape)
print("x_test Shapes ",x_test.shape)
print("y_test Shapes ",y_test.shape)

#---------------------------------------------------------------------------------------
"5.Data Classification  "
print("---------------------------------------------------------------------")
print("4.Data Classification --Unsupervised Machine Learning  ")
print("---------------------------------------------------------------------")

print("4.Data Classification --1.Random Forest Algorithm   ")
print("==================================================")

from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators = 10)  
rf.fit(x_train, y_train)
rf_prediction = rf.predict(x_test)


Result_1=accuracy_score(y_test, rf_prediction)*100
from sklearn.metrics import confusion_matrix

print()
print("-------------------------------------------------------")
print("Random Forest")
print()
print(metrics.classification_report(y_test,rf_prediction))
print()
print("Random Forest Accuracy is:",Result_1,'%')
print()
print("Random Forest Confusion Matrix:")
cm2=confusion_matrix(y_test, rf_prediction)
print(cm2)
print("-------------------------------------------------------")
print()
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm2, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
fpr, tpr, _ = metrics.roc_curve(y_test,  rf_prediction)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



#---------------------------------------------------------------------------------------------

print("4.Data Classification --2.Decision tree  Algorithm   ")
print("==================================================")

from sklearn.tree import DecisionTreeClassifier 
dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
dt.fit(x_train, y_train)
dt_prediction=dt.predict(x_test)
print()
print("-------------------------------------------------------")
print("Decision Tree")
print()
Result_2=accuracy_score(y_test, dt_prediction)*100
print(metrics.classification_report(y_test,dt_prediction))
print()
print("DT Accuracy is:",Result_2,'%')
print()
print("Decision Tree Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, dt_prediction)
print(cm1)
print("-------------------------------------------------------")
print()
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm1, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
#ROC graph
fpr, tpr, _ = metrics.roc_curve(y_test,  dt_prediction)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#------------------------------------------------------------------------------

"Xgboost Algorithm "


print("==================================================")


from numpy import loadtxt
from xgboost import XGBClassifier# fit model no training data
model = XGBClassifier()
model.fit(x_train, y_train)
model_prediction=model.predict(x_test)
print()
print("-------------------------------------------------------")
print("Xgboost   Algorithm  ")
print()
Result_3=accuracy_score(y_test, model_prediction)*100
print(metrics.classification_report(y_test,model_prediction))
print()
print("Xgboost Algorithm  Accuracy is:",Result_3,'%')
print()
print("Xgboost Algorithm  Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, model_prediction)
print(cm1)
print("-------------------------------------------------------")
print()
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm1, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
#ROC graph
fpr, tpr, _ = metrics.roc_curve(y_test,  model_prediction)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#----------------------------------------------------------------
print("5.Data Prediction")
print("==================================================")


import matplotlib.pyplot as plt

labels = ['Random Forest ', 'DT', 'Xgboost ']
data_a = [Result_1, Result_2, Result_3]

plt.figure(figsize=(8, 6))
plt.bar(labels, data_a, color=['blue', 'green','red'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Different Models')
plt.ylim(0, 1)  # Set y-axis limit to better visualize the data
plt.show()

from easygui import *
Key = "Enter the Quantum  Id to be Search"
  
# window title
title = "Quantum Computing  system  Id "
# creating a integer box
str_to_search1 = enterbox(Key, title)
input = int(str_to_search1)

import tkinter as tk
if (model_prediction[input] ==0 ):
    print("NON ATTACKED")
    root = tk.Tk()
    T = tk.Text(root, height=20, width=30)
    T.pack()
    T.insert(tk.END, "Non ATTACKED ")
    tk.mainloop()
elif (model_prediction[input] ==1 ):
    print("ATTACKED ")
    root = tk.Tk()
    T = tk.Text(root, height=20, width=30)
    T.pack()
    T.insert(tk.END, "ATTACKED ")
    tk.mainloop()