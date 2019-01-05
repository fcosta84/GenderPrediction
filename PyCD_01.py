"""
@author: 748309
"""

from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

#[height, weight, shoe size]
X = [[181,80,44], [177,70,43], [160,60,38], [154,54,37], [166,65,40], [190,90,47], [175,64,39],
     [177,70,40], [159,55,37], [171,75,42], [181,85,43]]

#Gender
Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']


#Decision Tree Classifier
dt_clf = tree.DecisionTreeClassifier()
dt_clf.fit(X,Y)
dt_prediction = dt_clf.predict([[190,70,43]])
print(dt_prediction)


#Support Vector Classifier
sv_clf = SVC()
sv_clf.fit(X,Y)
sv_prediction = sv_clf.predict([[190,70,43]])
print(sv_prediction)


#Gaussian Naive Bayes Classifier
nb_clf = GaussianNB()
nb_clf.fit(X,Y)
nb_prediction = nb_clf.predict([[190,70,43]])
print(nb_prediction)
