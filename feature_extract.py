import os
import math
from os import walk
import pandas as pd
import numpy as np
import time
from datetime import timedelta


file_names = []
dir_names = []

FILE_DIRECTORY =  os.getcwd() + "\\"
SisFall_ALL_DIRECTORY =  FILE_DIRECTORY + "SisFall_dataset\\"

for (dirpath, dirnames, filenames) in walk(SisFall_ALL_DIRECTORY):
    file_names.extend(filenames)
df_Files_Trials = pd.DataFrame({"File": file_names})



def compute_trial_file(trial_file_name):

    df_measurement = pd.DataFrame(pd.read_csv(trial_file_name, header = None, sep = ',', 
                                         names = ["ax1", "ay1", "az1", 
                                                  "gx1", "gy1", "gz1",
                                                  "ax2", "ay2", "az2"], 
                                                   skiprows= 0))

    # 1st ,2nd and 3rd column is the acceleration data measured by the sensor ADXL345.   Resolution: 13 bits / Range: +-16g
    # 4th ,5th ,6th column is the rotation data measured by the sensor ITG3200.          Resolution: 16 bits / Range: +-2000°/s
    # 7th ,8th ,9th column is the acceleration data measured by the sensor MMA8451Q.     Resolution: 14 bits / Range: +-8g
    
    """Data are in bits with the following characteristics:
    In order to convert the acceleration data (AD) given in bits into gravity, use this equation: 
    Acceleration [g]: [(2*Range)/(2^Resolution)]*AD
    In order to convert the rotation data (RD) given in bits into angular velocity, use this equation:
    Angular velocity [°/s]: [(2*Range)/(2^Resolution)]*RD
    """
   
    Sensor1_Resolution = 13
    Sensor1_Range = 16
    acc = (2*Sensor1_Range/2**Sensor1_Resolution)

   
    Sensor2_Resolution = 16
    Sensor2_Range = 2000
    gyr = (2*Sensor2_Range/2**Sensor2_Resolution)

 
    k = 0
    
    for i in range(0, len(df_measurement) - 256, 128):
        w = 256  # Size of sliding window (256 points at 200Hz = 1.25 seconds)
        w1 = df_measurement.iloc[i:i + w][:]
        

        fn_ax = lambda row: (acc*row.ax1)
        col = w1.apply(fn_ax, axis=1) 
        w1 = w1.assign(ax=col.values)

        fn_ay = lambda row: (acc*row.ay1)
        col = w1.apply(fn_ay, axis=1) 
        w1 = w1.assign(ay=col.values)

        fn_az = lambda row: (acc*row.az1)
        col = w1.apply(fn_az, axis=1) 
        w1 = w1.assign(az=col.values)

        
        #S1_N_XYZ                                                                           signal magnitude vector(SVM)
        fn_svm = lambda row: math.sqrt((row.ax)**2 + (row.ay)**2 + (row.az)**2) 
        col = w1.apply(fn_svm, axis=1) 
        w1 = w1.assign(S1_N_XYZ=col.values)
        

        # S1_N_HOR
        fn_hor = lambda row: math.sqrt((row.ay)**2 + (row.az)**2) 
        col = w1.apply(fn_hor, axis=1) 
        w1 = w1.assign(S1_N_HOR=col.values)
        

        # S1_N_VER
        fn_ver = lambda row: math.sqrt((row.ax)**2 + (row.az)**2) 
        col = w1.apply(fn_ver, axis=1) 
        w1 = w1.assign(S1_N_VER=col.values)
        

        SVM_max = w1["S1_N_XYZ"].max()              
        SVM_min = w1["S1_N_XYZ"].min()
        diff_SVM = SVM_max - SVM_min

        corr_val_NV = w1["S1_N_XYZ"].corr(w1["S1_N_VER"])
        corr_val_NH = w1["S1_N_XYZ"].corr(w1["S1_N_HOR"])
        corr_val_HV = w1["S1_N_HOR"].corr(w1["S1_N_VER"])


        corr_val_xy = w1["ax"].corr(w1["ay"])     #Correlation coefficeint
        corr_val_xz = w1["ax"].corr(w1["az"])
        corr_val_yz = w1["ay"].corr(w1["az"])

        df_Features = pd.DataFrame({"SVM_max": [SVM_max],
                                    "SVM_min": [SVM_min],
                                    "diff_SVM": [diff_SVM],
                                    "corr_val_NV": [corr_val_NV],
                                    "corr_val_NH": [corr_val_NH],
                                    "corr_val_HV" :[corr_val_HV],
                                    "corr_val_xy":[corr_val_xy],
                                    "corr_val_xz":[corr_val_xz],
                                    "corr_val_yz":[corr_val_yz]})
       
        
        field_name = "ax"  #SRA, Frequency Center 

        variance = w1[field_name].var()
        mean = w1[field_name].mean()
        median = w1[field_name].median()
        maximum = w1[field_name].max()
        minimum = w1[field_name].min()
        rang = maximum - minimum
        skewness = w1[field_name].skew()
        Energy = (w1[field_name]**2).sum()

        
        std = w1[field_name].std()                         #Standard Deviation
        kurtosis = w1[field_name].kurtosis()               #Kurtosis
        RMS = math.sqrt(np.mean(w1[field_name]**2))        #Root Mean Square
        
        df_Features_2 = pd.DataFrame({"variance_" + field_name:[variance],
                                      "mean_" + field_name:[mean],
                                      "median_" + field_name:[median],
                                      "maximum_" + field_name:[maximum],
                                      "minimum_" + field_name:[minimum],
                                      "rang_" + field_name:[rang],
                                      "skewness_" + field_name:[skewness],
                                      "Energy_" + field_name:[Energy],
                                      "std_" + field_name:[std], 
                                      "kurtosis_" + field_name:[kurtosis],
                                      "RMS_" + field_name:[RMS]})                              



        df_Features =  pd.concat([df_Features, df_Features_2], axis=1)

        

        field_name = "ay" #SRA, RMS Frequency


        variance = w1[field_name].var()
        mean = w1[field_name].mean()
        median = w1[field_name].median()
        maximum = w1[field_name].max()
        minimum = w1[field_name].min()
        rang = maximum - minimum
        RMS = math.sqrt(np.mean(w1[field_name]**2))
        Energy = (w1[field_name]**2).sum()
        

        std = w1[field_name].std()                         #Standard Deviation
        skewness =  w1[field_name].skew()                  #skewness
        kurtosis = w1[field_name].kurtosis()               #Kurtosis

        df_Features_2 = pd.DataFrame({"variance_" + field_name:[variance],
                                      "mean_" + field_name:[mean],
                                      "median_" + field_name:[median],
                                      "maximum_" + field_name:[maximum],
                                      "minimum_" + field_name:[minimum],
                                      "rang_" + field_name:[rang],
                                      "RMS_" + field_name:[RMS],
                                      "Energy_" + field_name:[Energy],
                                      "std_" + field_name:[std],
                                      "skewness_" + field_name:[skewness],
                                      "kurtosis_" + field_name:[kurtosis]})   


        df_Features =  pd.concat([df_Features, df_Features_2], axis=1)
        


        field_name = "az" #SRA,Frequency Center


        variance = w1[field_name].var()
        median = w1[field_name].median()
        maximum = w1[field_name].max()
        minimum = w1[field_name].min()
        rang = maximum - minimum
        kurtosis = w1[field_name].kurtosis()
        RMS = math.sqrt(np.mean(w1[field_name]**2))
        

        mean = w1[field_name].mean()                       #Mean
        std = w1[field_name].std()                         #Standard Deviation
        skewness =  w1[field_name].skew()                  #skewness
        Energy = (w1[field_name]**2).sum()

        df_Features_2 = pd.DataFrame({"variance_" + field_name:[variance],
                                      "median_" + field_name:[median],
                                      "maximum_" + field_name:[maximum],
                                      "minimum_" + field_name:[minimum],
                                      "rang_" + field_name:[rang],
                                      "kurtosis_" + field_name:[kurtosis],
                                      "RMS_" + field_name:[RMS],
                                      "mean_" + field_name:[mean],
                                      "std_" + field_name:[std],
                                      "skewness_" + field_name:[skewness],
                                      "Energy_" + field_name:[Energy]})   
        

        df_Features =  pd.concat([df_Features, df_Features_2], axis=1)
        
        
        
        trial_file_name = row['File']

        df_Features_2 = pd.DataFrame({"Fall_ADL": [trial_file_name[0:1]],
                                      "Age_Cat": [trial_file_name[4:6]],
                                      "Act_Type": [trial_file_name[0:3]],
                                      "Trial_no.": [trial_file_name[9:12]],})  

        df_Features =  pd.concat([df_Features_2, df_Features], axis=1)
        
        
        
        #These lines are for creating the file structure:
        #df_field_list = pd.DataFrame(list(df_Features.columns)).T
        #df_field_list.to_csv(FILE_DIRECTORY + 'ADL_Falls.csv', mode='w', header=False)

        # writes the record/instance data:
        outfile = open(FILE_DIRECTORY + "ADL_Falls.csv", 'a')
        df_Features.to_csv(outfile, header=False)
        outfile.close()


        del df_Features
        del df_Features_2
    
       
        

file_list = df_Files_Trials[["File"]]
total_num_iter = len(file_list)
iter_no = 1


for index, row in file_list.iterrows():
    
    my_data_file_name = SisFall_ALL_DIRECTORY + row['File']
    print("ITERATION NO: " + str(iter_no) + "/" + str(total_num_iter))
    iter_no +=1

    
    # USE THIS CONDITION (IF) to control processing scope. 
    if iter_no >4501: 
        print("SKIPPING TRIAL FILE: " + row['File'])
        continue
    
    print("PROCESSING TRIAL FILE: " + row['File'])



    compute_trial_file(my_data_file_name)
