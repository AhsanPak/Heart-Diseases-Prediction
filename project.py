import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("heart.csv")


def tran_cat_to_num(df):
    if df['trestbps'] >= 120:
        return 1
    else:
        return 0
    
df['trestbps']=df.apply(tran_cat_to_num,axis=1)

y = df.target.values
x_data = df.drop(['target'], axis = 1)




x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=2)
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T




#RandomForest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(x_train.T, y_train.T)
resultRF=rf.score(x_test.T,y_test.T)*100
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(resultRF))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train.T, y_train.T)
resultDT=dtc.score(x_test.T, y_test.T)*100
print("Decision Tree Test Accuracy {:.2f}%".format(resultDT))


#Naive Bayes

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train.T, y_train.T)
resultNB=nb.score(x_test.T,y_test.T)*100
print("Accuracy of Naive Bayes: {:.2f}%".format(resultNB))


# KNN Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(x_train.T, y_train.T)
prediction = knn.predict(x_test.T)
print("{} NN Score: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))
# try ro find best k value
scoreList = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(x_train.T, y_train.T)
    scoreList.append(knn2.score(x_test.T, y_test.T))
 

plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

resultKNN=max(scoreList)*100
print("Maximum KNN Score is {:.2f}%".format(resultKNN))















