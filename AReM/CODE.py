#%%
import warnings
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
#%% [markdown]
# b. Read data as test and training data

#%%
testData_1 = pd.read_csv('AReM/bending1/dataset1.csv',skiprows=4,error_bad_lines=False)
testData_2 = pd.read_csv('AReM/bending1/dataset2.csv',skiprows=4,error_bad_lines=False)
testData_3 = pd.read_csv('AReM/bending2/dataset1.csv',skiprows=4,error_bad_lines=False)
testData_4 = pd.read_csv('AReM/bending2/dataset2.csv',skiprows=4,error_bad_lines=False)
testData_5 = pd.read_csv('AReM/cycling/dataset1.csv',skiprows=4,error_bad_lines=False)
testData_6 = pd.read_csv('AReM/cycling/dataset2.csv',skiprows=4,error_bad_lines=False)
testData_7 = pd.read_csv('AReM/cycling/dataset3.csv',skiprows=4,error_bad_lines=False)
testData_8 = pd.read_csv('AReM/lying/dataset1.csv',skiprows=4,error_bad_lines=False)
testData_9 = pd.read_csv('AReM/lying/dataset2.csv',skiprows=4,error_bad_lines=False)
testData_10 = pd.read_csv('AReM/lying/dataset3.csv',skiprows=4,error_bad_lines=False)
testData_11 = pd.read_csv('AReM/sitting/dataset1.csv',skiprows=4,error_bad_lines=False)
testData_12 = pd.read_csv('AReM/sitting/dataset2.csv',skiprows=4,error_bad_lines=False)
testData_13 = pd.read_csv('AReM/sitting/dataset3.csv',skiprows=4,error_bad_lines=False)
testData_14 = pd.read_csv('AReM/standing/dataset1.csv',skiprows=4,error_bad_lines=False)
testData_15 = pd.read_csv('AReM/standing/dataset2.csv',skiprows=4,error_bad_lines=False)
testData_16 = pd.read_csv('AReM/standing/dataset3.csv',skiprows=4,error_bad_lines=False)
testData_17 = pd.read_csv('AReM/walking/dataset1.csv',skiprows=4,error_bad_lines=False)
testData_18 = pd.read_csv('AReM/walking/dataset2.csv',skiprows=4,error_bad_lines=False)
testData_19 = pd.read_csv('AReM/walking/dataset3.csv',skiprows=4,error_bad_lines=False)
#%%
trainData_1 = pd.read_csv('AReM/bending1/dataset3.csv',skiprows=4,error_bad_lines=False)
trainData_2 = pd.read_csv('AReM/bending1/dataset4.csv',skiprows=4,error_bad_lines=False)
trainData_3 = pd.read_csv('AReM/bending1/dataset5.csv',skiprows=4,error_bad_lines=False)
trainData_4 = pd.read_csv('AReM/bending1/dataset6.csv',skiprows=4,error_bad_lines=False)
trainData_5 = pd.read_csv('AReM/bending1/dataset7.csv',skiprows=4,error_bad_lines=False)
trainData_6 = pd.read_csv('AReM/bending2/dataset3.csv',skiprows=4,error_bad_lines=False)
trainData_7 = pd.read_csv('AReM/bending2/dataset4.csv',skiprows=4,error_bad_lines=False)
trainData_8 = pd.read_csv('AReM/bending2/dataset5.csv',skiprows=4,error_bad_lines=False)
trainData_9 = pd.read_csv('AReM/bending2/dataset6.csv',skiprows=4,error_bad_lines=False)
trainData_10 = pd.read_csv('AReM/cycling/dataset4.csv',skiprows=4,error_bad_lines=False)
trainData_11 = pd.read_csv('AReM/cycling/dataset5.csv',skiprows=4,error_bad_lines=False)
trainData_12 = pd.read_csv('AReM/cycling/dataset6.csv',skiprows=4,error_bad_lines=False)
trainData_13 = pd.read_csv('AReM/cycling/dataset7.csv',skiprows=4,error_bad_lines=False)
trainData_14 = pd.read_csv('AReM/cycling/dataset8.csv',skiprows=4,error_bad_lines=False)
trainData_15 = pd.read_csv('AReM/cycling/dataset9.csv',skiprows=4,error_bad_lines=False)
trainData_16 = pd.read_csv('AReM/cycling/dataset10.csv',skiprows=4,error_bad_lines=False)
trainData_17 = pd.read_csv('AReM/cycling/dataset11.csv',skiprows=4,error_bad_lines=False)
trainData_18 = pd.read_csv('AReM/cycling/dataset12.csv',skiprows=4,error_bad_lines=False)
trainData_19 = pd.read_csv('AReM/cycling/dataset13.csv',skiprows=4,error_bad_lines=False)
trainData_20 = pd.read_csv('AReM/cycling/dataset14.csv',skiprows=4,error_bad_lines=False)
trainData_21 = pd.read_csv('AReM/cycling/dataset15.csv',skiprows=4,error_bad_lines=False)
trainData_22 = pd.read_csv('AReM/lying/dataset4.csv',skiprows=4,error_bad_lines=False)
trainData_23 = pd.read_csv('AReM/lying/dataset5.csv',skiprows=4,error_bad_lines=False)
trainData_24 = pd.read_csv('AReM/lying/dataset6.csv',skiprows=4,error_bad_lines=False)
trainData_25 = pd.read_csv('AReM/lying/dataset7.csv',skiprows=4,error_bad_lines=False)
trainData_26 = pd.read_csv('AReM/lying/dataset8.csv',skiprows=4,error_bad_lines=False)
trainData_27 = pd.read_csv('AReM/lying/dataset9.csv',skiprows=4,error_bad_lines=False)
trainData_28 = pd.read_csv('AReM/lying/dataset10.csv',skiprows=4,error_bad_lines=False)
trainData_29 = pd.read_csv('AReM/lying/dataset11.csv',skiprows=4,error_bad_lines=False)
trainData_30 = pd.read_csv('AReM/lying/dataset12.csv',skiprows=4,error_bad_lines=False)
trainData_31 = pd.read_csv('AReM/lying/dataset13.csv',skiprows=4,error_bad_lines=False)
trainData_32 = pd.read_csv('AReM/lying/dataset14.csv',skiprows=4,error_bad_lines=False)
trainData_33 = pd.read_csv('AReM/lying/dataset15.csv',skiprows=4,error_bad_lines=False)
trainData_34 = pd.read_csv('AReM/sitting/dataset4.csv',skiprows=4,error_bad_lines=False)
trainData_35 = pd.read_csv('AReM/sitting/dataset5.csv',skiprows=4,error_bad_lines=False)
trainData_36 = pd.read_csv('AReM/sitting/dataset6.csv',skiprows=4,error_bad_lines=False)
trainData_37 = pd.read_csv('AReM/sitting/dataset7.csv',skiprows=4,error_bad_lines=False)
trainData_38 = pd.read_csv('AReM/sitting/dataset8.csv',skiprows=4,error_bad_lines=False)
trainData_39 = pd.read_csv('AReM/sitting/dataset9.csv',skiprows=4,error_bad_lines=False)
trainData_40 = pd.read_csv('AReM/sitting/dataset10.csv',skiprows=4,error_bad_lines=False)
trainData_41 = pd.read_csv('AReM/sitting/dataset11.csv',skiprows=4,error_bad_lines=False)
trainData_42 = pd.read_csv('AReM/sitting/dataset12.csv',skiprows=4,error_bad_lines=False)
trainData_43 = pd.read_csv('AReM/sitting/dataset13.csv',skiprows=4,error_bad_lines=False)
trainData_44 = pd.read_csv('AReM/sitting/dataset14.csv',skiprows=4,error_bad_lines=False)
trainData_45 = pd.read_csv('AReM/sitting/dataset15.csv',skiprows=4,error_bad_lines=False)
trainData_46 = pd.read_csv('AReM/standing/dataset4.csv',skiprows=4,error_bad_lines=False)
trainData_47 = pd.read_csv('AReM/standing/dataset5.csv',skiprows=4,error_bad_lines=False)
trainData_48 = pd.read_csv('AReM/standing/dataset6.csv',skiprows=4,error_bad_lines=False)
trainData_49 = pd.read_csv('AReM/standing/dataset7.csv',skiprows=4,error_bad_lines=False)
trainData_50 = pd.read_csv('AReM/standing/dataset8.csv',skiprows=4,error_bad_lines=False)
trainData_51 = pd.read_csv('AReM/standing/dataset9.csv',skiprows=4,error_bad_lines=False)
trainData_52 = pd.read_csv('AReM/standing/dataset10.csv',skiprows=4,error_bad_lines=False)
trainData_53 = pd.read_csv('AReM/standing/dataset11.csv',skiprows=4,error_bad_lines=False)
trainData_54 = pd.read_csv('AReM/standing/dataset12.csv',skiprows=4,error_bad_lines=False)
trainData_55 = pd.read_csv('AReM/standing/dataset13.csv',skiprows=4,error_bad_lines=False)
trainData_56 = pd.read_csv('AReM/standing/dataset14.csv',skiprows=4,error_bad_lines=False)
trainData_57 = pd.read_csv('AReM/standing/dataset15.csv',skiprows=4,error_bad_lines=False)
trainData_58 = pd.read_csv('AReM/walking/dataset4.csv',skiprows=4,error_bad_lines=False)
trainData_59 = pd.read_csv('AReM/walking/dataset5.csv',skiprows=4,error_bad_lines=False)
trainData_60 = pd.read_csv('AReM/walking/dataset6.csv',skiprows=4,error_bad_lines=False)
trainData_61 = pd.read_csv('AReM/walking/dataset7.csv',skiprows=4,error_bad_lines=False)
trainData_62 = pd.read_csv('AReM/walking/dataset8.csv',skiprows=4,error_bad_lines=False)
trainData_63 = pd.read_csv('AReM/walking/dataset9.csv',skiprows=4,error_bad_lines=False)
trainData_64 = pd.read_csv('AReM/walking/dataset10.csv',skiprows=4,error_bad_lines=False)
trainData_65 = pd.read_csv('AReM/walking/dataset11.csv',skiprows=4,error_bad_lines=False)
trainData_66 = pd.read_csv('AReM/walking/dataset12.csv',skiprows=4,error_bad_lines=False)
trainData_67 = pd.read_csv('AReM/walking/dataset13.csv',skiprows=4,error_bad_lines=False)
trainData_68 = pd.read_csv('AReM/walking/dataset14.csv',skiprows=4,error_bad_lines=False)
trainData_69 = pd.read_csv('AReM/walking/dataset15.csv',skiprows=4,error_bad_lines=False)
#%% [markdown]
# c(1). Time domain features

#%% [markdown]
# From my research, the time domain features have: mean, max, min, range, standard deviation, kurtosis and skewness.

#%% [markdown]
# c(2). The time-domain features minimum, maximum, mean, median, standard deviation, first quartile, and third quartile for all of the 6 time series in each instance

#%%
row0 = ["min1","max1","mean1","median1","std1","1st_quart1","3rd_quart1","min2","max2","mean2","median2","std2","1st_quart2","3rd_quart2","min3","max3","mean3","median3","std3","1st_quart3","3rd_quart3","min4","max4","mean4","median4","std4","1st_quart4","3rd_quart4","min5","max5","mean5","median5","std5","1st_quart5","3rd_quart5","min6","max6","mean6","median6","std6","1st_quart6","3rd_quart6"]
dataFrame = pd.DataFrame(columns=row0)

#%%
data_arr = [testData_1, testData_2, testData_3, testData_4, testData_5, testData_6, testData_7, testData_8, testData_9, testData_10, testData_11, testData_12, testData_13, testData_14, testData_15, testData_16, testData_17, testData_18, testData_19,trainData_1,trainData_2,trainData_3,trainData_4,trainData_5,trainData_6,trainData_7,trainData_8,trainData_9,trainData_10,trainData_11,trainData_12,trainData_13,trainData_14,trainData_15,trainData_16,trainData_17,trainData_18,trainData_19,trainData_20,trainData_21,trainData_22,trainData_23,trainData_24,trainData_25,trainData_26,trainData_27,trainData_28,trainData_29,trainData_30,trainData_31,trainData_32,trainData_33,trainData_34,trainData_35,trainData_36,trainData_37,trainData_38,trainData_39,trainData_40,trainData_41,trainData_42,trainData_43,trainData_44,trainData_45,trainData_46,trainData_47,trainData_48,trainData_49,trainData_50,trainData_51,trainData_52,trainData_53,trainData_54,trainData_55,trainData_56,trainData_57,trainData_58,trainData_59,trainData_60,trainData_61,trainData_62,trainData_63,trainData_64,trainData_65,trainData_66,trainData_67,trainData_68,trainData_69]
for i in range(0,len(data_arr)):
    row = [] 
    mini = data_arr[i].min()
    maxi = data_arr[i].max() 
    mean = np.mean(data_arr[i])
    median = data_arr[i].median()
    stad_d = np.std(data_arr[i])   
    first_quart = data_arr[i].quantile(.25)
    third_quart = data_arr[i].quantile(.75)

    for t in range(1,7):
        row.append(mini[t])
        row.append(maxi[t])
        row.append(mean[t])
        row.append(median[t])
        row.append(stad_d[t])
        row.append(first_quart[t])
        row.append(third_quart[t])
    dataFrame.loc[i]=row
dataFrame
#%% [markdown]
# c(3). Bootstrapped to standard-deviation
#%%
id_min = np.arange(0,42,1)
for i in id_min:
    std_data = dataFrame.iloc[:,i].values
    print(row0[i],bs.bootstrap(std_data, stat_func=bs_stats.std, alpha=0.1))

#%% [markdown]
# c(4). 3 important features

#%% [markdown]
# From my judgement, I think the mean, median and standard deviation are the most important 3 features

#%% [markdown]
# d(1). Plot mean, median and std to time-series 1,2 and 6

#%%
train_data = dataFrame.iloc[19:,[2,3,4,9,10,11,37,38,39]]
train_data.insert(9,'class',['bending','bending','bending','bending','bending','bending','bending','bending','bending','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others'])
sb.pairplot(train_data, hue='class')
#%% [markdown]
# d(2). Plot mean, median and std to time-series 1,2 and 12
#%%
row0 = ["mean1","median1","std1","mean2","median2","std2"]
train_df1 = pd.DataFrame(columns=row0)
row1 = ["mean12","median12","std12"]
train_df2 = pd.DataFrame(columns=row1)
#%%
train_arr = [trainData_1,trainData_2,trainData_3,trainData_4,trainData_5,trainData_6,trainData_7,trainData_8,trainData_9,trainData_10,trainData_11,trainData_12,trainData_13,trainData_14,trainData_15,trainData_16,trainData_17,trainData_18,trainData_19,trainData_20,trainData_21,trainData_22,trainData_23,trainData_24,trainData_25,trainData_26,trainData_27,trainData_28,trainData_29,trainData_30,trainData_31,trainData_32,trainData_33,trainData_34,trainData_35,trainData_36,trainData_37,trainData_38,trainData_39,trainData_40,trainData_41,trainData_42,trainData_43,trainData_44,trainData_45,trainData_46,trainData_47,trainData_48,trainData_49,trainData_50,trainData_51,trainData_52,trainData_53,trainData_54,trainData_55,trainData_56,trainData_57,trainData_58,trainData_59,trainData_60,trainData_61,trainData_62,trainData_63,trainData_64,trainData_65,trainData_66,trainData_67,trainData_68,trainData_69]
for i in range(0,len(train_arr)):
    row = [] 
    mean = np.mean(train_arr[i].iloc[:240,:])
    median = train_arr[i].iloc[:240,:].median()
    stad_d = np.std(train_arr[i].iloc[:240,:])   

    for t in [1,2]:
        row.append(mean[t])
        row.append(median[t])
        row.append(stad_d[t])
    train_df1.loc[i]=row

    row=[]
    mean = np.mean(train_arr[i].iloc[240:,:])
    median = train_arr[i].iloc[240:,:].median()
    stad_d = np.std(train_arr[i].iloc[240:,:])
    row.append(mean[6])
    row.append(median[6])
    row.append(stad_d[6])
    train_df2.loc[i]=row    
#%%
train_df = pd.concat([train_df1, train_df2],axis=1)
train_df.insert(9,'class',['bending','bending','bending','bending','bending','bending','bending','bending','bending','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others','others'])
sb.pairplot(train_df,hue='class')
#%% [markdown]
# There is no big difference in these 2 figures.

#%% [markdown]
# d(3). RFE & logistic regression, calculate p-values

#%%
row0 = ["min1","max1","mean1","median1","std1","1st_quart1","3rd_quart1","min2","max2","mean2","median2","std2","1st_quart2","3rd_quart2","min3","max3","mean3","median3","std3","1st_quart3","3rd_quart3","min4","max4","mean4","median4","std4","1st_quart4","3rd_quart4","min5","max5","mean5","median5","std5","1st_quart5","3rd_quart5","min6","max6","mean6","median6","std6","1st_quart6","3rd_quart6"]
train_df = pd.DataFrame(columns=row0)
train_df_1 = pd.DataFrame()
train_df_2 = pd.DataFrame()
train_df_3 = pd.DataFrame()
train_df_4 = pd.DataFrame()
train_df_5 = pd.DataFrame()
train_df_6 = pd.DataFrame()
train_df_7 = pd.DataFrame()
train_df_8 = pd.DataFrame()
train_df_9 = pd.DataFrame()
train_df_10 = pd.DataFrame()
train_df_11 = pd.DataFrame()
train_df_12 = pd.DataFrame()
train_df_13 = pd.DataFrame()
train_df_14 = pd.DataFrame()
train_df_15 = pd.DataFrame()
train_df_16 = pd.DataFrame()
train_df_17 = pd.DataFrame()
train_df_18 = pd.DataFrame()
train_df_19 = pd.DataFrame()
train_df_20 = pd.DataFrame()
#%%
# Creat the dataset we need
L = np.arange(1,21,1)
train_df_arr = [train_df_1,train_df_2,train_df_3,train_df_4,train_df_5,train_df_6,train_df_7,train_df_8,train_df_9,train_df_10,train_df_11,train_df_12,train_df_13,train_df_14,train_df_15,train_df_16,train_df_17,train_df_18,train_df_19,train_df_20]
train_arr = [trainData_1,trainData_2,trainData_3,trainData_4,trainData_5,trainData_6,trainData_7,trainData_8,trainData_9,trainData_10,trainData_11,trainData_12,trainData_13,trainData_14,trainData_15,trainData_16,trainData_17,trainData_18,trainData_19,trainData_20,trainData_21,trainData_22,trainData_23,trainData_24,trainData_25,trainData_26,trainData_27,trainData_28,trainData_29,trainData_30,trainData_31,trainData_32,trainData_33,trainData_34,trainData_35,trainData_36,trainData_37,trainData_38,trainData_39,trainData_40,trainData_41,trainData_42,trainData_43,trainData_44,trainData_45,trainData_46,trainData_47,trainData_48,trainData_49,trainData_50,trainData_51,trainData_52,trainData_53,trainData_54,trainData_55,trainData_56,trainData_57,trainData_58,trainData_59,trainData_60,trainData_61,trainData_62,trainData_63,trainData_64,trainData_65,trainData_66,trainData_67,trainData_68,trainData_69]

for l in L:
    for m in range(1,l+1):   
        for i in range(0,len(train_arr)):       
            row=[]  
            mean = np.mean(train_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:])
            median = train_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:].median()
            stad_d = np.std(train_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:])
            mini = train_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:].min()
            maxi = train_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:].max() 
            first_quart = train_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:].quantile(.25)
            third_quart = train_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:].quantile(.75)
            for t in range(1,7):
                row.append(mini[t])
                row.append(maxi[t])
                row.append(mean[t])
                row.append(median[t])
                row.append(stad_d[t])
                row.append(first_quart[t])
                row.append(third_quart[t])
            train_df.loc[i]=row
        
        train_df.insert(42, 'class', [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] )
        train_df_arr[l-1] = train_df_arr[l-1].append(train_df)
        # train_df_arr[l-1] = pd.concat([train_df_arr[l-1],train_df], axis=1)
        train_df = pd.DataFrame(columns=row0)
        
    # train_df_arr[l-1].insert(int(42*l),'class',[1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

#%%
# Calculate pvalues
import statsmodels.api as sm
for l in L:
    X = np.asarray(train_df_arr[l-1].drop(['class'], axis=1))
    y = np.asarray(train_df_arr[l-1]['class'])
    logit = sm.Logit(y, X)
    result = logit.fit(method='bfgs', maxiter=500)
    print(l, result.pvalues[:])

#%% [markdown]
# those 'nan's appear because when we don't cut the dataset enough, they will not have enough time-domain data for us to calculate the p-value
#%% [markdown]
# d(3). Calculate RFE
#%%
# change the dataset into the format we need to calculate RFE
row0 = ["min1","max1","mean1","median1","std1","1st_quart1","3rd_quart1","min2","max2","mean2","median2","std2","1st_quart2","3rd_quart2","min3","max3","mean3","median3","std3","1st_quart3","3rd_quart3","min4","max4","mean4","median4","std4","1st_quart4","3rd_quart4","min5","max5","mean5","median5","std5","1st_quart5","3rd_quart5","min6","max6","mean6","median6","std6","1st_quart6","3rd_quart6"]
train_df = pd.DataFrame(columns=row0)
train_df_1 = pd.DataFrame()
train_df_2 = pd.DataFrame()
train_df_3 = pd.DataFrame()
train_df_4 = pd.DataFrame()
train_df_5 = pd.DataFrame()
train_df_6 = pd.DataFrame()
train_df_7 = pd.DataFrame()
train_df_8 = pd.DataFrame()
train_df_9 = pd.DataFrame()
train_df_10 = pd.DataFrame()
train_df_11 = pd.DataFrame()
train_df_12 = pd.DataFrame()
train_df_13 = pd.DataFrame()
train_df_14 = pd.DataFrame()
train_df_15 = pd.DataFrame()
train_df_16 = pd.DataFrame()
train_df_17 = pd.DataFrame()
train_df_18 = pd.DataFrame()
train_df_19 = pd.DataFrame()
train_df_20 = pd.DataFrame()
#%%
L = np.arange(1,21,1)
train_df_arr = [train_df_1,train_df_2,train_df_3,train_df_4,train_df_5,train_df_6,train_df_7,train_df_8,train_df_9,train_df_10,train_df_11,train_df_12,train_df_13,train_df_14,train_df_15,train_df_16,train_df_17,train_df_18,train_df_19,train_df_20]
train_arr = [trainData_1,trainData_2,trainData_3,trainData_4,trainData_5,trainData_6,trainData_7,trainData_8,trainData_9,trainData_10,trainData_11,trainData_12,trainData_13,trainData_14,trainData_15,trainData_16,trainData_17,trainData_18,trainData_19,trainData_20,trainData_21,trainData_22,trainData_23,trainData_24,trainData_25,trainData_26,trainData_27,trainData_28,trainData_29,trainData_30,trainData_31,trainData_32,trainData_33,trainData_34,trainData_35,trainData_36,trainData_37,trainData_38,trainData_39,trainData_40,trainData_41,trainData_42,trainData_43,trainData_44,trainData_45,trainData_46,trainData_47,trainData_48,trainData_49,trainData_50,trainData_51,trainData_52,trainData_53,trainData_54,trainData_55,trainData_56,trainData_57,trainData_58,trainData_59,trainData_60,trainData_61,trainData_62,trainData_63,trainData_64,trainData_65,trainData_66,trainData_67,trainData_68,trainData_69]

for l in L:
    for m in range(1,l+1):   
        for i in range(0,len(train_arr)):       
            row=[]  
            mean = np.mean(train_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:])
            median = train_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:].median()
            stad_d = np.std(train_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:])
            mini = train_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:].min()
            maxi = train_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:].max() 
            first_quart = train_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:].quantile(.25)
            third_quart = train_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:].quantile(.75)
            for t in range(1,7):
                row.append(mini[t])
                row.append(maxi[t])
                row.append(mean[t])
                row.append(median[t])
                row.append(stad_d[t])
                row.append(first_quart[t])
                row.append(third_quart[t])
            train_df.loc[i]=row
        
        # train_df.insert(42, 'class', [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] )
        # train_df_arr[l-1] = train_df_arr[l-1].append(train_df)
        train_df_arr[l-1] = pd.concat([train_df_arr[l-1],train_df], axis=1)
        train_df = pd.DataFrame(columns=row0)
        
    train_df_arr[l-1].insert(int(42*l),'class',[1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#%% 
# Calculate RFE
lp_score = {}
best_features_df = {}
for l in L:
    X = train_df_arr[l-1].drop(['class'], axis=1)
    y = train_df_arr[l-1]['class']
    clf = LogisticRegression()
    warnings.simplefilter(action='ignore', category=FutureWarning)
    rfecv = RFECV(estimator=clf, cv=StratifiedKFold(5), scoring='accuracy')
    rfecv.fit(X, y)
    lp_score[l] = rfecv.grid_scores_[rfecv.n_features_ - 1]
    # print(rfecv.n_features_)
    feature = []
    for index, col in enumerate(X.columns):
        if rfecv.support_[index] == True:
           feature.append(col) 
    best_features_df[l] = feature


#%%
# Show the best (l,p)
print(lp_score)
print('bestL =', max(lp_score, key=lp_score.get), '\n', 'best features are:', best_features_df[max(lp_score, key=lp_score.get)])

#%% [markdown]
# The wrong way to do this cross-validation is to use simple K-fold, beacuse the datasets are time series data, we have to make sure the folded data still keeps its order.
# The right way is what we are doing noe, split data by order and do the K-fold, which is now stratified 5-fold.
#%% [markdown]
# d(4). Show more result for the classification we did

#%%
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

x_train = train_df_arr[max(lp_score, key=lp_score.get)-1][best_features_df[max(lp_score, key=lp_score.get)]]
y_train = train_df_arr[max(lp_score, key=lp_score.get)-1]['class']

clf = LogisticRegression()
clf.fit(x_train, y_train)
con_matrix = confusion_matrix(y_train, clf.predict(x_train))
print(con_matrix)
x_train
#%%
fpr, tpr, thresholds = roc_curve(y_train, clf.predict(x_train))
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='yellow')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve (auc = %0.2f)' % roc_auc)
plt.show()

#%% 
logit = sm.Logit(y_train, x_train)
result = logit.fit_regularized()
print(result.params)

#%%
print(result.pvalues)

#%% [markdown]
# d(5). Test clf on test data

#%%
# creat test dataset
row0 = ["min1","max1","mean1","median1","std1","1st_quart1","3rd_quart1","min2","max2","mean2","median2","std2","1st_quart2","3rd_quart2","min3","max3","mean3","median3","std3","1st_quart3","3rd_quart3","min4","max4","mean4","median4","std4","1st_quart4","3rd_quart4","min5","max5","mean5","median5","std5","1st_quart5","3rd_quart5","min6","max6","mean6","median6","std6","1st_quart6","3rd_quart6"]
temp = pd.DataFrame(columns=row0)
test_df = pd.DataFrame()

#%%
test_arr = [testData_1,testData_2,testData_3,testData_4,testData_5,testData_6,testData_7,testData_8,testData_9,testData_10,testData_11,testData_12,testData_13,testData_14,testData_15,testData_16,testData_17,testData_18,testData_19]
for m in range(1,4):
    for i in range(0,len(test_arr)):       
        row=[]  
        mean = np.mean(test_arr[i].iloc[int((m-1)*160):int(m*160),:])
        median = test_arr[i].iloc[int((m-1)*160):int(m*160),:].median()
        stad_d = np.std(test_arr[i].iloc[int((m-1)*160):int(m*160),:])
        mini = test_arr[i].iloc[int((m-1)*160):int(m*160),:].min()
        maxi = test_arr[i].iloc[int((m-1)*160):int(m*160),:].max() 
        first_quart = test_arr[i].iloc[int((m-1)*160):int(m*160),:].quantile(.25)
        third_quart = test_arr[i].iloc[int((m-1)*160):int(m*160),:].quantile(.75)
        for t in range(1,7):
            row.append(mini[t])
            row.append(maxi[t])
            row.append(mean[t])
            row.append(median[t])
            row.append(stad_d[t])
            row.append(first_quart[t])
            row.append(third_quart[t])
        temp.loc[i]=row
    
    test_df = pd.concat([test_df,temp], axis=1)
    temp = pd.DataFrame(columns=row0)
        
test_df.insert(126,'class',[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

#%%
x_test = test_df[best_features_df[max(lp_score, key=lp_score.get)]]
y_test = test_df['class']
clf.fit(x_test, y_test)
con_matrix = confusion_matrix(y_test, clf.predict(x_test))
print(con_matrix)
#%%
fpr, tpr, thresholds = roc_curve(y_test, clf.predict(x_test))
roc_auc = auc(fpr, tpr)
print(roc_auc)
#%% [markdown]
# we use l=3, which is the best l we get from the training data, to calculate the confusion matrix and roc_auc
# the result shows that when we cut data into 3 part, both test and train error is 0, they all classify every instance correctly

#%% [markdown]
# d(6).Do your classes seem to be well-separated to cause instability in calculating logistic regression parameters?

#%% [markdown]
# 

#%% [markdown]
# d(7).  A logistic regression model based on case-control sampling and adjust its parameter

#%%
# We choose to upsample minority class because there are only 9 of them, and the total sample is 69, witch is not large enough to downsample
from sklearn.utils import resample
train_df_3_minority = x_train.iloc[:9]
train_df_3_majority = x_train.iloc[9:]
train_df_3_minority_upsampled = resample(train_df_3_minority, replace=True, n_samples=60, random_state=123) 
train_df_3_upsampled = pd.concat([train_df_3_minority_upsampled, train_df_3_majority])
train_df_3_upsampled.insert(12, 'class', [1]*60+[0]*60)
#%%
X = train_df_3_upsampled.drop('class', axis=1)
y = train_df_3_upsampled['class']
clf.fit(X, y)
con_matrix = confusion_matrix(y, clf.predict(X))
print(con_matrix)

#%%
fpr, tpr, thresholds = roc_curve(y, clf.predict(X))
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='yellow')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve (auc = %0.2f)' % roc_auc)
plt.show()

#%% [markdown]
# e(1). Repeat 1(d)iii using L1-penalized logistic regression

#%%
# creat test dataset
test_df_1 = pd.DataFrame()
test_df_2 = pd.DataFrame()
test_df_3 = pd.DataFrame()
test_df_4 = pd.DataFrame()
test_df_5 = pd.DataFrame()
test_df_6 = pd.DataFrame()
test_df_7 = pd.DataFrame()
test_df_8 = pd.DataFrame()
test_df_9 = pd.DataFrame()
test_df_10 = pd.DataFrame()
test_df_11 = pd.DataFrame()
test_df_12 = pd.DataFrame()
test_df_13 = pd.DataFrame()
test_df_14 = pd.DataFrame()
test_df_15 = pd.DataFrame()
test_df_16 = pd.DataFrame()
test_df_17 = pd.DataFrame()
test_df_18 = pd.DataFrame()
test_df_19 = pd.DataFrame()
test_df_20 = pd.DataFrame()
temp = pd.DataFrame(columns=row0)
L = np.arange(1,21,1)
test_arr = [testData_1,testData_2,testData_3,testData_4,testData_5,testData_6,testData_7,testData_8,testData_9,testData_10,testData_11,testData_12,testData_13,testData_14,testData_15,testData_16,testData_17,testData_18,testData_19]
test_df_arr = [test_df_1,test_df_2,test_df_3,test_df_4,test_df_5,test_df_6,test_df_7,test_df_8,test_df_9,test_df_10,test_df_11,test_df_12,test_df_13,test_df_14,test_df_15,test_df_16,test_df_17,test_df_18,test_df_19,test_df_20]

for l in L:
    for m in range(1,l+1):   
        for i in range(0,len(test_arr)):       
            row=[]  
            mean = np.mean(test_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:])
            median = test_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:].median()
            stad_d = np.std(test_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:])
            mini = test_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:].min()
            maxi = test_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:].max() 
            first_quart = test_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:].quantile(.25)
            third_quart = test_arr[i].iloc[int((m-1)*(480/l)):int(m*(480/l)),:].quantile(.75)
            for t in range(1,7):
                row.append(mini[t])
                row.append(maxi[t])
                row.append(mean[t])
                row.append(median[t])
                row.append(stad_d[t])
                row.append(first_quart[t])
                row.append(third_quart[t])
            temp.loc[i]=row
        
        test_df_arr[l-1] = pd.concat([test_df_arr[l-1],temp], axis=1)
        temp = pd.DataFrame(columns=row0)
        
    test_df_arr[l-1].insert(int(42*l),'class',[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#%%
from sklearn.linear_model import LogisticRegressionCV
l_score = {}
best_features_df = {}
for l in L:
    X = train_df_arr[l-1].drop(['class'], axis=1)
    y = train_df_arr[l-1]['class']
    clf = LogisticRegressionCV(penalty='l1',max_iter=500,cv=5,solver='liblinear')
    clf.fit(X,y)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    x_test = test_df_arr[l-1].drop(['class'], axis=1)
    y_test = pd.DataFrame()
    y_test.insert(0,'class', [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    l_score[l] = clf.score(x_test, y_test)


#%%
print(l_score)
print('bestL =', max(l_score, key=l_score.get))

#%% [markdown]
# e(2). Compair l1-penalty with p-value selection
#%% [markdown]
# Use l1 penalty is easier and in this case, it gives a better result.

#%% [markdown]
# f(1). Multiple class regression
#%%
l_score = {}
for l in L:
    x_train = train_df_arr[l-1].drop(['class'], axis=1)
    y_train = pd.DataFrame()
    y_train.insert(0,'class', [1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6])
    clf = LogisticRegressionCV(penalty='l1', multi_class='multinomial', solver='saga', cv=5)
    clf.fit(x_train, y_train)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    x_test = test_df_arr[l-1].drop(['class'], axis=1)
    y_test = pd.DataFrame()
    y_test.insert(0,'class', [1,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6])
    l_score[l] = clf.score(x_test, y_test)

#%%
print(l_score)
print('bestL =', max(l_score, key=l_score.get))
#%% [markdown]
# calculate confusion matrix

#%%
x_train = train_df_arr[max(l_score, key=l_score.get)-1].drop(['class'], axis=1)
x_test = test_df_arr[max(l_score, key=l_score.get)-1].drop(['class'], axis=1)
clf = LogisticRegressionCV(penalty='l1', multi_class='multinomial', solver='saga', cv=5).fit(x_train,y_train)
con_matrix = confusion_matrix(y_test, clf.predict(x_test))
print(con_matrix)

#%% 
# roc_auc, data creat
y_train1 = pd.DataFrame()
y_train1.insert(0,'class', [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
y_train2 = pd.DataFrame()
y_train2.insert(0,'class', [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
y_train3 = pd.DataFrame()
y_train3.insert(0,'class', [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
y_train4 = pd.DataFrame()
y_train4.insert(0,'class', [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
y_train5 = pd.DataFrame()
y_train5.insert(0,'class', [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
y_train6 = pd.DataFrame()
y_train6.insert(0,'class', [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])
y_test1 = pd.DataFrame()
y_test1.insert(0,'class', [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
y_test2 = pd.DataFrame()
y_test2.insert(0,'class', [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
y_test3 = pd.DataFrame()
y_test3.insert(0,'class', [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0])
y_test4 = pd.DataFrame()
y_test4.insert(0,'class', [0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0])
y_test5 = pd.DataFrame()
y_test5.insert(0,'class', [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0])
y_test6 = pd.DataFrame()
y_test6.insert(0,'class', [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1])
#%%
y_train_arr = [y_train1, y_train2, y_train3, y_train4, y_train5, y_train6]
y_test_arr = [y_test1, y_test2, y_test3, y_test4, y_test5, y_test6]
color = ['yellow', 'green', 'red', 'orange', 'gray', 'pink']
clf = LogisticRegressionCV(cv=StratifiedKFold(5),penalty="l1", solver='liblinear')
plt.figure()
for i in range(0,6):
    clf.fit(x_train, y_train_arr[i])
    fpr, tpr, thresholds = roc_curve(y_test_arr[i], clf.predict(x_test))
    roc_auc = auc(fpr, tpr)   
    plt.plot(fpr, tpr, color=color[i], label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve (AUC in the label)')
plt.legend(loc="lower right")
plt.show()
print('The order from up to bottom on the figure label is: bending, cycling, lying, sitting, standing, walking')
#%% [markdown]
# f(2). Naive Bayes Classifer (Gaussian)

#%%
# use guide from: https://stackoverflow.com/questions/16379313/how-to-use-the-a-k-fold-cross-validation-in-scikit-with-naive-bayes-classifier-a
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
l_score = {}
for l in L:
    x_train = train_df_arr[l-1].drop(['class'], axis=1)
    y_train = pd.DataFrame()
    y_train.insert(0,'class', [1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6])
    clf = GaussianNB()
    # clf.fit(x_train, y_train)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    l_score[l] = np.mean(cross_val_score(clf, x_train, y_train, cv=5))
#%%
print(l_score)
print('bestL =', max(l_score, key=l_score.get))
#%% [markdown]
# f(2). Naive Bayes Classifer (Multinomial)

#%%
from sklearn.naive_bayes import MultinomialNB
l_score = {}
for l in L:
    x_train = train_df_arr[l-1].drop(['class'], axis=1)
    y_train = pd.DataFrame()
    y_train.insert(0,'class', [1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6])
    clf = MultinomialNB()
    warnings.simplefilter(action='ignore', category=FutureWarning)
    l_score[l] = np.mean(cross_val_score(clf, x_train, y_train, cv=5))

#%%
print(l_score)
print('bestL =', max(l_score, key=l_score.get))

#%% [markdown]
# In this dataset, GaussianNB performs better than MultinomialNB

#%% [markdown]
# f(3). Which method is better for multi-class classification

#%% [markdown]
# From thest 3 test we did in question f, using logistic regression with "mult-class = multinomial" has the best accuracy. It should be the best model in this problem.

#%% [markdown]
# ISLR 3.7.4
#%% [markdown]
# 1. It is hard to define which training RSS will be lower. However, as the true relationship between X and Y is linear, I believe the linear moedel will have a greater chance to have a lower trainning RSS.

# 2. For the test RSS, we still don't have enough evidence to judge which method will be lower. But the polynomial model is more likely to be overfitted, so I guess linear model will still be lower.

# 3. As the true relationship is not linear anymore, and the calculation of RSS is sum(estimate y - y), the polynomial regression will have a lower training RSS because it will be more flexable than linear one.

# 4. For the test data, we don't have enough information to make the decision. It depends on how close the true relationship is to linear, if it is close enough, maybe linear regression will perform better, otherwise it will have a higher RSS.
#%% [markdown]
# ISLR 4.7.3

#%%

#%% [markdown]
# ISLR 4.7.7
