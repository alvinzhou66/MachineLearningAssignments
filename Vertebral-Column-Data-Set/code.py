# To add a new cell, type '#%%'

# To add a new markdown cell, type '#%% [markdown]'

#%%

from IPython import get_ipython

#%% [markdown]

# # import packages

#%%

import pandas as pd

import numpy as np

import seaborn as sb

import matplotlib.pyplot as plt

#%% [markdown]

# # read file (use weka to tansfer it into .csv)

#%%

file = pd.read_csv("file.csv")
file

#%% [markdown]

# # b(i) scatterplot

#%%

# using guide from https://seaborn.pydata.org/generated/seaborn.pairplot.html

sb.pairplot(file, hue="class")

#%% [markdown]

# # b(ii) boxplot

#%%

sb.boxplot(x="pelvic_incidence", y="class", data=file)

#%%

sb.boxplot(x="pelvic_tilt", y="class", data=file)

#%%

sb.boxplot(x="lumbar_lordosis_angle", y="class", data=file)

#%%

sb.boxplot(x="sacral_slope", y="class", data=file)

#%%

sb.boxplot(x="pelvic_radius", y="class", data=file)

#%%

sb.boxplot(x="degree_spondylolisthesis", y="class", data=file)

#%% [markdown]

# # b(iii) divide data into training & testing

#%%

class1 = pd.read_csv("class1.csv")

class0 = pd.read_csv("class0.csv")

#"Normal" = Class0, "Abnormal" = class1

training_data = pd.concat([class1[0:140],class0[0:70]])

training_data

#%%

testing_data = pd.concat([class1[140:], class0[70:]])

testing_data

#%%

#set training x,y and testing x,y

x_train = np.array(training_data[["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"]])

y_train = np.array(training_data["class"])

y_train = np.array([0 if clas == "Normal" else 1 for clas in y_train])



x_test = np.array(testing_data[["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"]])

y_test = np.array(testing_data["class"])

y_test = np.array([0 if clas == "Normal" else 1 for clas in y_test])

#%% [markdown]

# # c(i) KNN classifier

#%%

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

#%% [markdown]

# # c(ii) try different K and calculate parameters

#%%

K = np.arange(1,211,3)

training_error = {}

testing_error = {}

testing_error_array = []

training_error_array = []



for k in K:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train, y_train)

    testing_error_array.append(1-accuracy_score(y_test, knn.predict(x_test)))

    training_error_array.append(1-accuracy_score(y_train, knn.predict(x_train)))

    testing_error.update({k: 1-accuracy_score(y_test, knn.predict(x_test))})

    training_error.update({k: 1-accuracy_score(y_train, knn.predict(x_train))})



train_error_rate = []

del training_error[1]

train_error_rate.append(min(training_error.values()))



testing_error

# in this case, we have the lowest testing_error when K = 4, so 4 is the suitable K

#%%

training_error

#%%

plt.title("Test(red) and Train(green) error")

plt.plot(K, testing_error_array, color="red")

plt.plot(K, training_error_array, color="green")

plt.grid(color='grey', linestyle='--', linewidth=1,alpha=0.3)

#%%

# using functions from:https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

from sklearn.metrics import confusion_matrix

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(x_train, y_train)

confusion_matrix(y_test, knn.predict(x_test))



# using guide from:https://blog.csdn.net/u011956147/article/details/78967145

tn, fp, fn, tp = confusion_matrix(y_test, knn.predict(x_test)).ravel()

true_positive_rate = tp/(tp + fn)

true_negative_rate = tn/(tn + fp)

precision = tp/(tp + fp)

f_score = 2*(precision*true_positive_rate)/(precision+true_positive_rate)

print("TPR =",true_positive_rate,", TNR =",true_negative_rate,", PPV =",precision,", F-score =",f_score)

#%% [markdown]

# c(iii) plot best error rate, plot learning curve

#%%

best_test_error_rate = {}

best_train_error_rate = {}

N = np.arange(10, 211, 10)



for n in N:

    testing_error_array_c = []

    training_error_array_c = []

    training_data_c = pd.concat([class0[0:int(n/3)], class1[0:int(2*n/3)]])

    x_train_c = np.array(training_data_c[["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"]])

    y_train_c = np.array(training_data_c["class"])

    y_train_c = np.array([0 if clas == "Normal" else 1 for clas in y_train_c])



    for k in range(1, n, 5):

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(x_train_c, y_train_c)

        testing_error_array_c.append(1-accuracy_score(y_test, knn.predict(x_test)))

        training_error_array_c.append(1-accuracy_score(y_train_c, knn.predict(x_train_c)))



    best_test_error_rate.update({n:min(testing_error_array_c)})

    best_train_error_rate.update({n:min(training_error_array_c)})



train_error_rate.append(min(best_train_error_rate.values()))



#%%

best_train_error_rate

#%%

plt.title("Learning Curve")

plt.plot(N, tuple(best_test_error_rate.values()))

plt.grid(color='grey', linestyle='--', linewidth=1,alpha=0.3)

#%% [markdown]

# d. try different distance

#%%

# we can modify p in KNeighborsClassifier to apply different distance:https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

K = np.arange(1, 197, 5)

testing_error = {}

training_error = {}



for k in K:

    knn = KNeighborsClassifier(n_neighbors=k, p=1)

    knn.fit(x_train, y_train)

    testing_error.update({k: 1-accuracy_score(y_test, knn.predict(x_test))})

    training_error.update({k: 1-accuracy_score(y_train, knn.predict(x_train))})



del training_error[1]

train_error_rate.append(min(training_error.values()))



testing_error



#%%

training_error

#%%

# from above cell's calculate we have k*=26

log10P = [10**0.1, 10**0.2, 10**0.3, 10**0.4, 10**0.5, 10**0.6, 10**0.7, 10**0.8, 10**0.9, 10]

testing_error = {}

training_error = {}



for p in log10P:

    knn = KNeighborsClassifier(n_neighbors=26, p=p)

    knn.fit(x_train, y_train)

    testing_error.update({p:1-accuracy_score(y_test, knn.predict(x_test))})

    training_error.update({p:1-accuracy_score(y_train, knn.predict(x_train))})



train_error_rate.append(min(training_error.values()))



testing_error

#%%

training_error

#%%

# From above cell, we noticed that 4 log10P have the same lowest testing_error

# In this case, the best log10P are: log10(0.1), log10(0.2), log10(0.4) and log10(0.6)

# The best P are: 0.1, 0.2, 0.4 and 0.6

#%%

K = np.arange(1, 197, 5)

testing_error = {}

training_error = {}



for k in K:

    knn = KNeighborsClassifier(n_neighbors=k,metric="chebyshev")

    knn.fit(x_train, y_train)

    testing_error.update({k: 1-accuracy_score(y_test, knn.predict(x_test))})

    training_error.update({k: 1-accuracy_score(y_train, knn.predict(x_train))})



del training_error[1]

train_error_rate.append(min(training_error.values()))



testing_error

#%%

training_error

#%%

# In chebyshev distance, when k =16, we have the lowest test error

# k* = 16

#%%

# Figured out I have to use brute force in mahalanobis distance:https://github.com/scikit-learn/scikit-learn/issues/11807

K = np.arange(1, 197, 3)

testing_error = {}

training_error = {}



for k in K:

    knn = KNeighborsClassifier(n_neighbors=k,metric="mahalanobis",metric_params={"V": np.cov(x_train)},algorithm="brute")

    knn.fit(x_train, y_train)

    testing_error.update({k: 1-accuracy_score(y_test, knn.predict(x_test))})

    training_error.update({k: 1-accuracy_score(y_train, knn.predict(x_train))})



del training_error[1]

train_error_rate.append(min(training_error.values()))



testing_error

#%%

training_error

#%%

# When using mahalanobis distance, the lowest test error was on k = 1

# The result when interval=5 seems to be not good enough, so I tried in interval=3

# The new k* = 4

#%% [markdown]

# e. Weighted decision

#%%

K = np.arange(1, 197, 5)

testing_error = {}

training_error = {}



for k in K:

    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")

    knn.fit(x_train, y_train)

    testing_error.update({k: 1-accuracy_score(y_test, knn.predict(x_test))})

    training_error.update({k: 1-accuracy_score(y_train, knn.predict(x_train))})



del training_error[1]

train_error_rate.append(min(training_error.values()))



testing_error



#%%

training_error

#%%

# k* = 6

# best_test_error = 0.09999999999999998

#%%

K = np.arange(1, 197, 5)

testing_error = {}

training_error = {}



for k in K:

    knn = KNeighborsClassifier(n_neighbors=k, p=1, weights="distance")

    knn.fit(x_train, y_train)

    testing_error.update({k: 1-accuracy_score(y_test, knn.predict(x_test))})

    training_error.update({k: 1-accuracy_score(y_train, knn.predict(x_train))})



del training_error[1]

train_error_rate.append(min(training_error.values()))



testing_error

#%%

training_error

#%%

# k* = 26

# best_test_error = 0.09999999999999998

#%%

K = np.arange(1, 197, 5)

testing_error = {}

training_error = {}



for k in K:

    knn = KNeighborsClassifier(n_neighbors=k,metric="chebyshev", weights="distance")

    knn.fit(x_train, y_train)

    testing_error.update({k: 1-accuracy_score(y_test, knn.predict(x_test))})

    training_error.update({k: 1-accuracy_score(y_train, knn.predict(x_train))})



del training_error[1]

train_error_rate.append(min(training_error.values()))



testing_error

#%%

training_error

#%%

# k* = 16, 31, 36, 41, 61

# best_test_error = 0.10999999999999999

#%% [markdown]

# f. lowest training error rate

#%%

print(train_error_rate)

print("lowest train_error_rate =", min(train_error_rate))

#%%

