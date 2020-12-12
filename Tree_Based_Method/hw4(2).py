#%%
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
import warnings
warnings.filterwarnings('ignore')
#%% [markdown]
# b(1). Read data and fix the missing value

#%%
training_set = pd.read_csv('data/aps_failure_training_set.csv', skiprows=20)
test_set = pd.read_csv('data/aps_failure_test_set.csv', skiprows=20)
training_set = training_set.replace("na", "NaN")
test_set = test_set.replace("na","NaN")
#%%
x_training = training_set.drop('class', axis=1)
x_test = test_set.drop('class', axis=1)
y_training = training_set['class']
y_test = test_set['class']
mean_set = Imputer(missing_values="NaN", axis=1)
x_training = pd.DataFrame(mean_set.fit_transform(x_training), columns=x_training.columns)
x_training

#%%
mean_set = Imputer(missing_values="NaN", axis=1)
x_test = pd.DataFrame(mean_set.fit_transform(x_test),columns=x_test.columns)
x_test

#%% [markdown]
# b(2). Calculate CV for 170 features

#%%
coefficient_variations = {}

for i in x_training.columns:

    std = np.std(x_training[i])
    mean = np.mean(x_training[i])
    coefficient_variations[i] = std/mean

for key,value in coefficient_variations.items():
    print('{key}:{value}'.format(key = key, value = value))

#%% [markdown]
# b(3). Plot correlation
#%%
f = plt.figure(figsize=(10, 7))
plt.matshow(x_training.corr(), fignum=f.number)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=6)
plt.title('Correlation Matrix for training set', fontsize=16)

#%% [markdown]
# b(4). Pick highest CV features
#%%
# sqrt(171) round down = 13
high_cv_features = sorted(coefficient_variations, key=coefficient_variations.get)[-13:]
x_highCV = x_training[high_cv_features]
#%%
sb.pairplot(x_highCV)

#%%
sb.boxplot(x=x_highCV.iloc[:,0])

#%%
sb.boxplot(x=x_highCV.iloc[:,1])

#%%
sb.boxplot(x=x_highCV.iloc[:,2])

#%%
sb.boxplot(x=x_highCV.iloc[:,3])

#%%
sb.boxplot(x=x_highCV.iloc[:,4])

#%%
sb.boxplot(x=x_highCV.iloc[:,5])

#%%
sb.boxplot(x=x_highCV.iloc[:,6])

#%%
sb.boxplot(x=x_highCV.iloc[:,7])

#%%
sb.boxplot(x=x_highCV.iloc[:,8])

#%%
sb.boxplot(x=x_highCV.iloc[:,9])

#%%
sb.boxplot(x=x_highCV.iloc[:,10])

#%%
sb.boxplot(x=x_highCV.iloc[:,11])

#%%
sb.boxplot(x=x_highCV.iloc[:,12])

#%% [markdown]
# b(5). Determine the number of pos and neg class

#%%
print(training_set['class'].value_counts())
print('                                   ')
print('It is imbalanced')
#%% [markdown]
# c. Random Forest Classification
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import preprocessing
#%%
clf = RandomForestClassifier(random_state=0).fit(x_training,y_training)
cm0 = confusion_matrix(y_training, clf.predict(x_training))
print('Training Set Confusion_matrix:')
print('-----------------')
print(cm0)
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('Test Set Confusion_matrix:')
print('-----------------')
print(cm)
#%%
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_training)
y_pred0 = lb.fit_transform(clf.predict(x_training))
fpr, tpr, thresholds = roc_curve(y_train, y_pred0)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='yellow')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Training ROC curve (auc = %0.2f)' % roc_auc)
plt.show()
print('training auc =', roc_auc)
#%%
y_test1 = lb.fit_transform(y_test)
y_pred = lb.fit_transform(y_pred)
fpr, tpr, thresholds = roc_curve(y_test1, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='yellow')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test ROC curve (auc = %0.2f)' % roc_auc)
plt.show()
print('test auc =', roc_auc)
#%%
from sklearn.metrics import accuracy_score
misclassification_train = 1 - accuracy_score(y_training, clf.predict(x_training))
print('Training missclassification = ', misclassification_train)
misclassification_test = 1 - accuracy_score(y_test, clf.predict(x_test))
print('Test missclassification = ', misclassification_test)
#%%
clf = RandomForestClassifier(random_state=0, oob_score=True).fit(x_training,y_training)
print('out of bag error =', 1-clf.oob_score_)
print('test error =', 1-clf.score(x_test,y_test))

#%% [markdown]
# d. Handle imbalanced classes in Random Forest

#%%
# use guide from:https://chrisalbon.com/machine_learning/trees_and_forests/handle_imbalanced_classes_in_random_forests/
clf = RandomForestClassifier(random_state=0,class_weight='balanced').fit(x_training,y_training)
cm0 = confusion_matrix(y_training, clf.predict(x_training))
print('Training Set Confusion_matrix:')
print('-----------------')
print(cm0)
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('Test Set Confusion_matrix:')
print('-----------------')
print(cm)

#%%
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_training)
y_pred0 = lb.fit_transform(clf.predict(x_training))
fpr, tpr, thresholds = roc_curve(y_train, y_pred0)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='yellow')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Training ROC curve (auc = %0.2f)' % roc_auc)
plt.show()
print('training auc =', roc_auc)
#%%
y_test1 = lb.fit_transform(y_test)
y_pred = lb.fit_transform(y_pred)
fpr, tpr, thresholds = roc_curve(y_test1, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='yellow')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test ROC curve (auc = %0.2f)' % roc_auc)
plt.show()
print('test auc =', roc_auc)
#%%
misclassification_train = 1 - accuracy_score(y_training, clf.predict(x_training))
print('Training missclassification = ', misclassification_train)
misclassification_test = 1 - accuracy_score(y_test, clf.predict(x_test))
print('Test missclassification = ', misclassification_test)

#%%
clf = RandomForestClassifier(random_state=0, class_weight='balanced', oob_score=True).fit(x_training,y_training)
print('out of bag error =', 1-clf.oob_score_)
print('test error =', 1-clf.score(x_test,y_test))


#%% [markdown]
# e. Call weka to build a decision tree (10-fold CV)

#%%
import weka.core.converters as converters
import weka.core.jvm as jvm

#%%
training_weka = pd.concat([x_training[:500], y_training[:500]], axis=1, ignore_index=True)
training_weka.to_csv('training_weka.csv',index=False)
test_weka = pd.concat([x_test[:500], y_test[:500]], axis=1, ignore_index=True)
test_weka.to_csv('test_weka.csv', index=False)

#%%
jvm.start()

#%%
training_set = converters.load_any_file("training_weka.csv")
training_set.class_is_last()
test_set = converters.load_any_file("test_weka.csv")
test_set.class_is_last()

#%%
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random
import weka.plot.classifiers as plcls
#%%
clf = Classifier(classname="weka.classifiers.trees.LMT")
evl = Evaluation(training_set)
evl.crossvalidate_model(classifier=clf, data=training_set, num_folds = 10, rnd=Random(42))

#%%
print(evl.confusion_matrix)
print(evl.summary())
print(evl.class_details())

#%%
print('Training ROC')
plcls.plot_roc(evl, class_index=[0, 1], wait=True)

#%%
evl = Evaluation(test_set)
evl.crossvalidate_model(classifier=clf, data=test_set, num_folds = 10, rnd=Random(42))

#%%
print(evl.confusion_matrix)
print(evl.summary())
print(evl.class_details())

#%%
print('Training ROC')
plcls.plot_roc(evl, class_index=[0, 1], wait=True)

#%% [markdown]
# f.SMOTE the data first and then do e again

#%%
# using guide form:https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
x_training_0, y_training_0 = sm.fit_sample(x_training[:500],y_training.replace({0:'neg',1:'pos'})[:500]) 
training_smote = pd.concat([pd.DataFrame(x_training_0), pd.DataFrame(y_training_0)], axis=1, ignore_index=True)
training_smote.to_csv('training_smote.csv',index=False)
x_test_0, y_test_0 = sm.fit_sample(x_test[:500],y_test.replace({0:'neg', 1:'pos'})[:500])
test_smote = pd.concat([pd.DataFrame(x_test_0), pd.DataFrame(y_test_0)], axis=1, ignore_index=True)
test_smote.to_csv('test_smote.csv', index=False)
#%%
training_set = converters.load_any_file("training_smote.csv")
training_set.class_is_last()
test_set = converters.load_any_file("test_smote.csv")
test_set.class_is_last()

#%%
clf = Classifier(classname="weka.classifiers.trees.LMT")
evl = Evaluation(training_set)
evl.crossvalidate_model(classifier=clf, data=training_set, num_folds = 10, rnd=Random(42))

#%%
print(evl.confusion_matrix)
print(evl.summary())
print(evl.class_details())

#%%
print('Training ROC')
plcls.plot_roc(evl, class_index=[0, 1], wait=True)

#%%
evl = Evaluation(test_set)
evl.crossvalidate_model(classifier=clf, data=test_set, num_folds = 10, rnd=Random(42))

#%%
print(evl.confusion_matrix)
print(evl.summary())
print(evl.class_details())

#%%
print('Training ROC')
plcls.plot_roc(evl, class_index=[0, 1], wait=True)

#%%
