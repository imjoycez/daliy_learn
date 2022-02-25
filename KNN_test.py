from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np

# iris = load_iris()
# X,y = iris.data[:,:3], iris.target

data = np.loadtxt('E:/Joyce/Daily Work/20211105 Wheat powdery/data_classification120.csv',delimiter=",")
# split data into X and y
X, y = data[:,0:3], data[:,3]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=0)

# knn = KNeighborsClassifier(n_neighbors=3, weights="uniform")
knn = KNeighborsClassifier()

grid = {"n_neighbors":range(1,11,1),"weights":['uniform','distance']}
gs = GridSearchCV(estimator=knn, param_grid=grid, scoring="accuracy", n_jobs=-1, cv=5, verbose=10)
gs.fit(X_train,y_train)

print("best score:", gs.best_score_)
print("best params:", gs.best_params_)
print("best estimator:", gs.best_estimator_)

estimator = gs.best_estimator_
knn.fit(X_train,y_train)
y_hat = estimator.predict(X_test)
print(classification_report(y_test, y_hat))
