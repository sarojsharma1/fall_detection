import pandas as pd
import os
import random
import math
from numpy.random import permutation
from sklearn.metrics import confusion_matrix
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



FILE_DIRECTORY =  os.getcwd() + "\\"
my_data_file_name = FILE_DIRECTORY + "ADL_Falls.csv"

df_ADL_Falls = pd.DataFrame(pd.read_csv(my_data_file_name, sep = ','))

df_ADL_Falls.drop('0', axis=1, inplace=True)

df_only_ADLs = df_ADL_Falls[df_ADL_Falls.Fall_ADL == "D"]
df_only_Falls = df_ADL_Falls[df_ADL_Falls.Fall_ADL == "F"]

#print(df_only_ADLs.tail())
#print(df_only_Falls.tail())


# Randomly shuffle the index of each set (ADLs and Falls)


# First we prepare the sets of ADLs
random_indices = permutation(df_only_ADLs.index)
# Use a test-split (of 30% of the items)
test_split = math.floor(len(df_only_ADLs)*0.3)
# Test set with 30% of items
df_only_ADLs_test = df_only_ADLs.loc[random_indices[0:test_split]][0:3080]
# Train set with 70% of the items.
df_only_ADLs_train = df_only_ADLs.loc[random_indices[test_split:]][0:7200]


# Now we prepare the sets of Falls
random_indices = permutation(df_only_Falls.index)
# Use a test-split (of 30% of the items)
test_split = math.floor(len(df_only_Falls)*0.3)
# Test set with 30% of items
df_only_Falls_test = df_only_Falls.loc[random_indices[0:test_split]][0:3080]
# Train set with 70% of the items.
df_only_Falls_train = df_only_Falls.loc[random_indices[test_split:]][0:7200]


print("Total Falls: " + str(len(df_only_Falls)))
print("Total ADL: " + str(len(df_only_ADLs)))
print("GRAND Total: " + str(len(df_only_Falls)+len(df_only_ADLs)))
print("---------------------------------------")
print("Train Falls: "+ str(len(df_only_Falls_train)))
print("Train ADL: "+ str(len(df_only_ADLs_train)))
print("Train TOTAL: "+ str(len(df_only_ADLs_train)+len(df_only_Falls_train)))
print("---------------------------------------")
print("Test Falls: "+ str(len(df_only_Falls_test)))
print("Test ADL: "+ str(len(df_only_ADLs_test)))
print("Test TOTAL: "+ str(len(df_only_ADLs_test)+len(df_only_Falls_test)))
print("---------------------------------------")


# Prepare dataset with Test examples
frames = [df_only_Falls_test, df_only_ADLs_test]
df_ADL_Falls_test = pd.concat(frames)



# Prepare dataset with Train examples
frames = [df_only_Falls_train, df_only_ADLs_train]
df_ADL_Falls_train = pd.concat(frames)

'''
print("---------------------------------------")
print(df_ADL_Falls_train.shape)
print("---------------------------------------")
print(df_ADL_Falls_train.count())
print("---------------------------------------")
print(df_ADL_Falls_train.dtypes)


print("---------------------------------------")
print(df_ADL_Falls_test.shape)
print("---------------------------------------")
print(df_ADL_Falls_test.count())
print("---------------------------------------")
print(df_ADL_Falls_test.dtypes)
print("---------------------------------------")
'''




x_columns = ['SVM_max', 'SVM_min',
       'diff_SVM', 'corr_val_NV', 'corr_val_NH', 'corr_val_HV', 'corr_val_xy',
       'corr_val_xz', 'corr_val_yz', 'variance_ax', 'mean_ax', 'median_ax',
       'maximum_ax', 'minimum_ax', 'rang_ax', 'skewness_ax', 'Energy_ax',
       'std_ax', 'kurtosis_ax', 'RMS_ax', 'variance_ay', 'mean_ay',
       'median_ay', 'maximum_ay', 'minimum_ay', 'rang_ay', 'RMS_ay',
       'Energy_ay', 'std_ay', 'skewness_ay', 'kurtosis_ay', 'variance_az',
       'median_az', 'maximum_az', 'minimum_az', 'rang_az', 'kurtosis_az',
       'RMS_az', 'mean_az', 'std_az', 'skewness_az', 'Energy_az']


y_column = ['Fall_ADL']

X_train = df_ADL_Falls_train[x_columns]
y_train = (df_ADL_Falls_train[y_column]).values.ravel()

X_test = df_ADL_Falls_test[x_columns]
y_test = (df_ADL_Falls_test[y_column]).values.ravel()




#Feature selection and classification
for index in range(1,20):
    
    sel = RFE(GradientBoostingClassifier(n_estimators=100, random_state=4),n_features_to_select = index)
    sel.fit(X_train, y_train)
    
    X_train_rfe = sel.transform(X_train)
    X_test_rfe = sel.transform(X_test)
    
    feat = X_train.columns[sel.get_support()]
    print("no.of selected features=",index)
    print("selected features=",feat)

    # define the classifier
    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_rfe,y_train)
    
    y_predict = clf.predict(X_test_rfe)
    print("Prediction of test sets are",y_predict)


    #--->7.Evaluation(Results)
    print("accuracy is",accuracy_score(y_test,y_predict))
    print("---------------------------------")


    cm = confusion_matrix(y_test,y_predict, labels=["D", "F"])
    print("-----------------")
    print("Confusion Matrix:",cm)


    '''
    #Preview coeficients
    print(clf.coef_)         
    print(clf.intercept_)    


    n_TP = cm[1,1]
    n_FP = cm[1,0]
    n_TN = cm[0,0]
    n_FN = cm[0,1]


    print("---------------------------------------")
    # SENSITIVITY = TP / (TP + FN)
    svc_Sensitivity = n_TP / (n_TP + n_FN)
    print("svc_Sensitivity = ",svc_Sensitivity)

    # SPECIFICITY = TN / (FP + TN)
    svc_Specificity = n_TN / (n_FP + n_TN)
    print("svc_Specificity = ",svc_Specificity)

    # Precision = TP / (TP + FP)
    svc_Precision = n_TP / (n_TP + n_FP)
    print("svc_Precision = ",svc_Precision)

    # Accuracy = (TP + TN) / (TP + FP + TN + FN)
    svc_Accuracy = (n_TP + n_TN) / (n_TP + n_FP + n_TN + n_FN)
    print("svc_Accuracy = ",svc_Accuracy)
    '''
