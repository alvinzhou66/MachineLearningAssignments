#%%
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
import warnings
warnings.filterwarnings('ignore')

#%% [markdown]
# b. Read data and fix the missing value
#%%
columns = ["state","county","community","communityname", "fold","population","householdsize","racepctblack","racePctWhite","racePctAsian","racePctHisp","agePct12t21","agePct12t29","agePct16t24","agePct65up","numbUrban","pctUrban","medIncome","pctWWage","pctWFarmSelf","pctWInvInc","pctWSocSec","pctWPubAsst","pctWRetire","medFamInc","perCapInc","whitePerCap","blackPerCap","indianPerCap","AsianPerCap","OtherPerCap","HispPerCap","NumUnderPov","PctPopUnderPov","PctLess9thGrade","PctNotHSGrad","PctBSorMore","PctUnemployed","PctEmploy","PctEmplManu","PctEmplProfServ","PctOccupManu","PctOccupMgmtProf","MalePctDivorce","MalePctNevMarr","FemalePctDiv","TotalPctDiv","PersPerFam","PctFam2Par","PctKids2Par","PctYoungKids2Par","PctTeen2Par","PctWorkMomYoungKids","PctWorkMom","NumIlleg","PctIlleg","NumImmig","PctImmigRecent","PctImmigRec5","PctImmigRec8","PctImmigRec10","PctRecentImmig","PctRecImmig5","PctRecImmig8","PctRecImmig10","PctSpeakEnglOnly","PctNotSpeakEnglWell","PctLargHouseFam","PctLargHouseOccup","PersPerOccupHous","PersPerOwnOccHous","PersPerRentOccHous","PctPersOwnOccup","PctPersDenseHous","PctHousLess3BR","MedNumBR","HousVacant","PctHousOccup","PctHousOwnOcc","PctVacantBoarded","PctVacMore6Mos","MedYrHousBuilt","PctHousNoPhone","PctWOFullPlumb","OwnOccLowQuart","OwnOccMedVal","OwnOccHiQuart","RentLowQ","RentMedian","RentHighQ","MedRent","MedRentPctHousInc","MedOwnCostPctInc","MedOwnCostPctIncNoMtg","NumInShelters","NumStreet","PctForeignBorn","PctBornSameState","PctSameHouse85","PctSameCity85","PctSameState85","LemasSwornFT","LemasSwFTPerPop","LemasSwFTFieldOps","LemasSwFTFieldPerPop","LemasTotalReq","LemasTotReqPerPop","PolicReqPerOffic","PolicPerPop","RacialMatchCommPol","PctPolicWhite","PctPolicBlack","PctPolicHisp","PctPolicAsian","PctPolicMinor","OfficAssgnDrugUnits","NumKindsDrugsSeiz","PolicAveOTWorked","LandArea","PopDens","PctUsePubTrans","PolicCars","PolicOperBudg","LemasPctPolicOnPatr","LemasGangUnitDeploy","LemasPctOfficDrugUn","PolicBudgPerPop","ViolentCrimesPerPop"]
file = pd.read_csv('data/communities.data', names=columns)
training_set = file[0:1495]
test_set = file[1495:]
training_set = training_set.drop(["state","county","community","communityname","fold"], axis=1)
training_set = training_set.replace("?", "NaN")
# we use mean to fullfill missing values
mean_set = Imputer(missing_values="NaN", axis=1)
training_set = pd.DataFrame(mean_set.fit_transform(training_set), columns=training_set.columns)
training_set
#%%
test_set.reset_index(drop=True,inplace=True)
test_set = test_set.drop(["state","county","community","communityname","fold"], axis=1)
test_set = test_set.replace("?", "NaN")
# we use mean to fullfill missing values
mean_set = Imputer(missing_values="NaN", axis=1)
test_set = pd.DataFrame(mean_set.fit_transform(test_set), columns=test_set.columns)
test_set

#%% [markdown]
# c. Plot a correlation matrix

#%%
# use guide form: https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
f = plt.figure(figsize=(10, 7))
plt.matshow(training_set.corr(), fignum=f.number)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=6)
plt.title('Correlation Matrix for training set', fontsize=16)
#%% [markdown]
# d. Calculate the Coefficient of Variation

#%%
# to calculate CV for all features, we seprate features and label first
x_training_set = training_set.drop(['ViolentCrimesPerPop'], axis=1)
y_training_set = training_set['ViolentCrimesPerPop']
x_test_set = test_set.drop(['ViolentCrimesPerPop'], axis=1)
y_test_set = test_set['ViolentCrimesPerPop']
#%%
coefficient_variations = {}
for i in x_training_set.columns:
    std = np.std(x_training_set[i])
    mean = np.mean(x_training_set[i])
    coefficient_variations[i] = std/mean

for key,value in coefficient_variations.items():
    print('{key}:{value}'.format(key = key, value = value))

#%% [markdown]
# e(1). Make scatter plots and box plots for highest CVs

#%%
# sqrt(128) round down = 11
high_cv_features = sorted(coefficient_variations, key=coefficient_variations.get)[-11:]
x_highCV_set = x_training_set[high_cv_features]
sb.pairplot(x_highCV_set)
#%%
sb.boxplot(x=x_highCV_set.iloc[:,0])

#%%
sb.boxplot(x=x_highCV_set.iloc[:,1])

#%%
sb.boxplot(x=x_highCV_set.iloc[:,2])

#%%
sb.boxplot(x=x_highCV_set.iloc[:,3])

#%%
sb.boxplot(x=x_highCV_set.iloc[:,4])

#%%
sb.boxplot(x=x_highCV_set.iloc[:,5])

#%%
sb.boxplot(x=x_highCV_set.iloc[:,6])

#%%
sb.boxplot(x=x_highCV_set.iloc[:,7])

#%%
sb.boxplot(x=x_highCV_set.iloc[:,8])

#%%
sb.boxplot(x=x_highCV_set.iloc[:,9])

#%%
sb.boxplot(x=x_highCV_set.iloc[:,10])

#%% [markdown]
# e(2). Draw conclusion from the scatter plot and box plot

#%%

#%% [markdown]
# f. Use linear model on this dataset

#%%
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_training_set, y_training_set)
mse = mean_squared_error(y_test_set, reg.predict(x_test_set))
print('test MSE =', mse)
print('score =', reg.score(x_test_set, y_test_set))
#%% [markdown]
# g. Ridge regression model

#%%
from sklearn.linear_model import RidgeCV
reg = RidgeCV(cv=10).fit(x_training_set, y_training_set)
test_err = 1-reg.score(x_test_set, y_test_set)
print('test error =', test_err)
#%%
# h. Fit a LASSO model onto this dataset

#%%
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
reg = LassoCV(cv=10).fit(x_training_set, y_training_set)
test_err = 1-reg.score(x_test_set, y_test_set)
print('test error =', test_err)
# use guide from: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
best_features = []
feature = SelectFromModel(LassoCV(cv=10))
feature.fit(x_training_set, y_training_set)
for index, col in enumerate(x_training_set.columns):
    if feature.get_support()[index] == True:
        best_features.append(col) 
best_features

#%%
reg0 = LassoCV(cv=10, normalize=True)
reg0.fit(x_training_set,y_training_set)
test_err = 1-reg0.score(x_test_set, y_test_set)
print('test error =', test_err)

best_features0 = []
feature = SelectFromModel(reg0)
feature.fit(x_training_set, y_training_set)
for index, col in enumerate(x_training_set.columns):
    if feature.get_support()[index] == True:
        best_features0.append(col) 
best_features0

#%% [markdown]
# i. Fit a PCR model
#%%
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# use guide from: http://www.science.smith.edu/~jcrouser/SDS293/labs/lab11-py.html
pca = PCA()
x_training_set_reduced = pca.fit_transform(scale(x_training_set))
# 10-fold CV, with shuffle
n = len(x_training_set_reduced)
kf_10 = KFold( n_splits=10, shuffle=True, random_state=1)
reg = LinearRegression()
mse = []
# Calculate MSE with only the intercept (no principal components in regression)
score = -1*cross_val_score(reg, np.ones((n,1)), y_training_set.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()    
mse.append(score)
# Calculate MSE using CV for the 122 principle components, adding one component at the time.
for i in np.arange(1, 123):
    score = -1*cross_val_score(reg, x_training_set_reduced[:,:i], y_training_set.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
    mse.append(score)
print(mse.index(min(mse)))

#%%
x_test_set_reduced = pca.transform(scale(x_test_set))[:,:94]
reg.fit(x_training_set_reduced[:,:94], y_training_set)
mse = mean_squared_error(y_test_set, reg.predict(x_test_set_reduced))
print('test MSE =', mse)
#%% [markdown]
# j. Fit a Boost Tree and find out the best alpha

#%%
import xgboost
alpha = [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1]
mse = {}
for a in alpha:
    xgb_model = xgboost.XGBRegressor(cv=kf_10,reg_alpha=a).fit(x_training_set, y_training_set)
    y_pred = xgb_model.predict(x_test_set)
    mse[a] = mean_squared_error(y_test_set, y_pred)
mse
#%%
print(min(mse, key=mse.get),min(mse.values()))
#%%
