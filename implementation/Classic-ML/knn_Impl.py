from sklearn.datasets import load_iris
iris = load_iris()


import pandas as pd

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(iris.target)
iris_df['target'] = pd.Series(iris.target)  # 0, 1, 2 세가지의 소속 클래스를 갖는다.


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

X = iris_df.iloc[:, :4] # target열을 제외한 나머지 모두
y = iris_df.iloc[:, -1] # target열 (label로 사용됨)

def iris_knn(X, y, k):
  X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3)
  knn = KNeighborsClassifier(n_neighbors = k)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  return metrics.accuracy_score(y_test, y_pred)

# 데이터에 그대로
k = 3
scores = iris_knn(X, y, k)
print(f'n_neighbors가 {k:2d}일때 정확도: {scores:.3f}')

# train, test로 나눈 후
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3)
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
scores = metrics.accuracy_score(y_test, y_pred)
print(f'n_neighbors가 {k:2d}일때 정확도: {scores:.3f}')

scores = ( (y_test==y_pred).astype(int) ).sum() / len(y_test)
print(scores)



#########################################################################################
print('-'*100)


import numpy as np

INFINITE = 1000000000

## vector x의 모든 원소와 y 좌표 간에 거리를 계산하는 함수
def dist(x, y):
  result = (x-y)**2
  result = result[:,0] + result[:,1]
  return np.sqrt( result )

## X_test에 대한 예측값을 찾는 함수
def my_knn_fit_and_predict(n_neighbors, X_train, y_train, X_test):
  y_hat_test = []
  X_train = X_train.to_numpy()
  X_test = X_test.to_numpy()
  y_train = y_train.to_numpy()

  for i in range(len(X_test)):  # 하나의 loop동안 X_test 하나에 대한 label(예측 값)을 찾는다.
    tmp = np.zeros(len(X_test))
    distance_arr = dist(X_train, X_test[i])

    for k in range(n_neighbors):
      min_index = np.argmin(distance_arr) # 가장 작은 distance값을 갖는 index
      distance_arr[min_index] = INFINITE # 다음 번째로 작은 distance값의 index를 찾기 위해
      tmp[ y_train[ np.argmin(distance_arr) ] ] += 1 # tmp의 index는 label의 값을 tmp의 값은 label에 해당하는 빈도수를 나타낸다.

    label = np.argmax(tmp)
    y_hat_test.append(label)
  
  return np.array(y_hat_test)

y_pred = my_knn_fit_and_predict(3, X_train, y_train, X_test)
scores = metrics.accuracy_score(y_test, y_pred)
print(f'n_neighbors가 {k:2d}일때 정확도: {scores:.3f}')
