import pandas as pd
import os
import random
import math
from numpy.random import permutation
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score



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


print("Total no. of Falls datasets: " + str(len(df_only_Falls)))
print("Total no. of ADL(Activities of daily living) datasets: " + str(len(df_only_ADLs)))
print("Total no. of datasets: " + str(len(df_only_Falls)+len(df_only_ADLs)))
print("---------------------------------------")
print("Balanced datasets:")
print("Total no. of Falls Training datasets: "+ str(len(df_only_Falls_train)))
print("Total no. of ADL(Activities of daily living) Training datasets:"+ str(len(df_only_ADLs_train)))
print("Total no. of Training datasets: "+ str(len(df_only_ADLs_train)+len(df_only_Falls_train)))
print('\n')
print("Total no. of Falls Test datasets: "+ str(len(df_only_Falls_test)))
print("Total no. of ADL(Activities of daily living) Test datasets: "+ str(len(df_only_ADLs_test)))
print("Total no. of Test datasets: "+ str(len(df_only_ADLs_test)+len(df_only_Falls_test)))
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


x_columns = ['corr_val_NH','corr_val_HV', 'rang_ax', 'mean_ay', 'std_ay','kurtosis_ay', 'std_az','Energy_az','RMS_ax']


y_column = ['Fall_ADL']



X_train = df_ADL_Falls_train[x_columns]
y_train = (df_ADL_Falls_train[y_column]).values.ravel()

X_test = df_ADL_Falls_test[x_columns]
y_test = (df_ADL_Falls_test[y_column]).values.ravel()



# define the classifier
classifier = RandomForestClassifier(n_estimators=100)


print("Please Wait!!! Training is going on")
classifier.fit(X_train,y_train)

    
y_predict = classifier.predict(X_test)


#confusion matrix
cm = confusion_matrix(y_test,y_predict, labels=["D", "F"])
print("-----------------")
print("Confusion Matrix:",cm)

n_TP = cm[1,1]
n_FP = cm[1,0]
n_TN = cm[0,0]
n_FN = cm[0,1]


print("---------------------------------------")
# SENSITIVITY = TP / (TP + FN)
modal_Sensitivity = n_TP / (n_TP + n_FN)
print("modal_Sensitivity = ",modal_Sensitivity)

# SPECIFICITY = TN / (FP + TN)
modal_Specificity = n_TN / (n_FP + n_TN)
print("modal_Specificity = ",modal_Specificity)

# Precision = TP / (TP + FP)
modal_Precision = n_TP / (n_TP + n_FP)
print("modal_Precision = ",modal_Precision)

# Accuracy = (TP + TN) / (TP + FP + TN + FN)
modal_Accuracy = (n_TP + n_TN) / (n_TP + n_FP + n_TN + n_FN)
print("modal_Accuracy = ",modal_Accuracy)
    
    
#Preview coeficients
#print(classifier.coef_)         
#print(classifier.intercept_)    


#Save the model to disk
filename = 'train_file'
pickle.dump(classifier ,open(filename, 'wb'))







