# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:47:06 2019

@author: DEEPESH
"""

import warnings
warnings.filterwarnings("ignore")

#Including the header from python Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import load_model

#Function that is used to preprocess the NumberRecognitionBigger
def preprocessing_Number_data():
    all_vars = io.loadmat(r"D:\folder-assignment\NumberRecognitionBigger.mat")
    trainX = all_vars.get("X")
    label = all_vars.get("y")
    label=label.T
    train=[]
    complete_data =[]
    print("Conversion of DataFrom NumberRecognitionBiggerv.mat")
    imgLent = np.prod(trainX.shape[:-1])
    lengtht = trainX.shape[-1]
    train=trainX.reshape([imgLent, lengtht])
    train=train.T
    complete_data =np.append(train,label,axis=1)
   
    df=pd.DataFrame(complete_data)
    #CITATIONS:https://stackoverflow.com/questions/19851005/rename-pandas-dataframe-index
    df.rename(columns={784 : 'labels'},inplace=True)
    df.sort_values(by=['labels'], inplace=True)
    df.head(5)
    my_train_data =df.iloc[:,0:784]
    my_train_label=df.iloc[:,-1]
    my_train_data=np.array(my_train_data)
    my_train_label=np.array(my_train_label)
   
    return (my_train_data,my_train_label)


error_knn_1 =[]
score_knn_1 =[]

error_knn_5 =[]
score_knn_5 =[]

error_knn_10 =[]
score_knn_10 =[]


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,ReLU
from keras.layers import Dropout, Activation ,MaxPool2D,BatchNormalization
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import model_from_json
from keras import losses, optimizers


def CNN_model():
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Conv2D(5, kernel_size=3, input_shape=(28, 28, 1),
                   activation="linear", # only to match MATLAB defaults
                   data_format="channels_last"
                   ))

    model.add(ReLU())
    model.add(Flatten())
    model.add(
            Dense(units=10, activation="softmax") # 10 units, 10 digits
            ) # multiclass classification output, use softmax
   
    model.compile(optimizer=optimizers.SGD(momentum=0.9, lr=0.001),loss=losses.mean_squared_error,metrics=["accuracy"],)
   
    return model    

loss_cnn=[]
score_cnn=[]
error_cnn=[]

loss_cnn_4=[]
score_cnn_4=[]
error_cnn_4=[]


loss_ann=[]
score_ann=[]
error_ann=[]

from pathlib import Path
#
def predictions_csv():
    filefolder = Path(__file__).absolute().parent.absolute()
    SCRIPT_PATH = str((filefolder / "python_predict.py").absolute())
    with open(SCRIPT_PATH, "r") as file:
        script = file.read()
        exec(script)

def fit_predict_check_score_and_error(universalmodel,X_train,X_test,y_train,y_test):
    universalmodel.fit(X_train,y_train)
    prediction = universalmodel.predict(X_test)  
    return(metrics.accuracy_score(y_test,prediction),np.mean(y_test != prediction))

def Loop_for_computataion(my_train_data, my_train_label ,model,model_ann, status,question4_enable):
    flag=0
    Kfold_stratified_shuffleop = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for training_values, testing_values in Kfold_stratified_shuffleop.split(my_train_data, my_train_label):
        print("\n")
        print("TRAINING VALUES:", training_values, "TESTING VALUES:", testing_values)
        print("\n")

#using the standard naming convention X_train X_test,y_train,y_test
        X_train, X_test = my_train_data[training_values], my_train_data[testing_values]
        y_train, y_test = my_train_label[training_values], my_train_label[testing_values]
       
        print("KNN")
        knn = KNeighborsClassifier(n_neighbors=1)
        score,error=fit_predict_check_score_and_error(knn,X_train, X_test,y_train,y_test)       
        error_knn_1.append(error)
        score_knn_1.append(score)  

        knn = KNeighborsClassifier(n_neighbors=5)
        score,error=fit_predict_check_score_and_error(knn,X_train, X_test,y_train,y_test)        
        error_knn_5.append(error)        
        score_knn_5.append(score)
             
        knn = KNeighborsClassifier(n_neighbors=10)
        score,error=fit_predict_check_score_and_error(knn,X_train, X_test,y_train,y_test)        
        error_knn_10.append(error)
        score_knn_10.append(score)       
        
        if (question4_enable==1):
            print("RUNNING QUESTION 4 CNN")
            X_train = X_train.reshape(len(X_train),28,28,1)
            X_test = X_test.reshape(len(X_test),28,28,1)
            y_train = np_utils.to_categorical(y_train, 10)
            y_test = np_utils.to_categorical(y_test, 10)
            model.fit(X_train,y_train,batch_size=32,epochs=5)
            loss_4,accuracy_4=model.evaluate(X_test,y_test)
            loss_cnn_4.append(loss_4)
            score_cnn_4.append(accuracy_4)
            error_cnn_4.append(1-accuracy_4)
            flag=1
            filefolder = Path(__file__).absolute().parent.absolute()
            SCRIPT_PATH = str((filefolder / "python_predict.py").absolute())
            with open(SCRIPT_PATH, "r") as file:
                script = file.read()
                exec(script)
#            predictions_csv()
            
        else:
            print("performing computation")
                    
        if (status == 1):
            print("RUNNING BASIC CNN")
            cnn_split = list(StratifiedShuffleSplit(n_splits=2, test_size=0.1).split(X_train, y_train))
            idx_tr, idx_val = cnn_split[0]
       
            X_val, y_val = X_train[idx_val], y_train[idx_val]
            X_tr, y_tr = X_train[idx_tr], y_train[idx_tr]
       
            X_val = X_val.reshape(len(X_val),28,28,1)
            X_tr = X_tr.reshape(len(X_tr),28,28,1)
            X_test = X_test.reshape(len(X_test),28,28,1)
            y_test = np_utils.to_categorical(y_test, 10)  
            
            y_val = np_utils.to_categorical(y_val, 10)
            y_tr = np_utils.to_categorical(y_tr, 10)
          
           
            model.fit(X_tr, y_tr, validation_data=(X_val, y_val))
            print("Loss and accuracy and accuracy for Test CNN with Stratified Kfold ")
#            model_cnn.predict(X_test)
            loss,accuracy=model.evaluate(X_test,y_test)
            loss_cnn.append(loss)
            score_cnn.append(accuracy)
            error_cnn.append(1-accuracy)   
            
        else:
            if(flag!=1):
                y_train = np_utils.to_categorical(y_train,4)
                y_test = np_utils.to_categorical(y_test,4)
                model_ann.fit(X_train, y_train, epochs=5,verbose=1)
                print("Loss accuracy and error for Test ANN with Stratified Kfold")
#                model_ann.predict(X_test)
                loss_a,accuracy_a=model_ann.evaluate(X_test,y_test)
                loss_ann.append(loss_a)
                score_ann.append(accuracy_a)
                error_ann.append(1-accuracy_a)


    print("LOSS ACCURACY AND ERROR MEAN OF CNN MODEL FOR K FOLD VALUE 5 ")
    print("------------------QUESTION4-----------------------------------")
    print(np.mean(loss_cnn_4))
    print("ACCURACY")
    print(np.mean(score_cnn_4))
    print("ERROR")
    print(np.mean(error_cnn_4))
    
    
    print("LOSS ACCURACY AND ERROR MEAN OF CNN MODEL FOR K FOLD VALUE 5")
    print(np.mean(loss_cnn))
    print("ACCURACY")
    print(np.mean(score_cnn))
    print("ERROR")
    print(np.mean(error_cnn))
    
    print("LOSS ACCURACY AND ERROR OF MEAN ANN MODEL FOR K FOLD VALUE 5")
    print("LOSS")
    print(np.mean(loss_ann))
    print("ACCURACY")
    print(np.mean(score_ann))
    print("ERROR")
    print(np.mean(error_ann))
                       
                               
    print("#########KNN ERROR FOR K=1##############")
    print(np.mean(error_knn_1))
    print("#########KNN ACCURACY FOR K=1##############")
    print(np.mean(score_knn_1))  

    print("#########KNN ERROR FOR K=5##############")
    print(np.mean(error_knn_5))
    print("#########KNN ACCURACY FOR K=5##############")
    print(np.mean(score_knn_5))    

    print("#########KNN ERROR FOR K=10##############")
    print(np.mean(error_knn_10))
    print("#########KNN ACCURACY FOR K=10##############")
    print(np.mean(score_knn_10))

                     
######################################################################
##############################QUESTION 1#########################
######################################################################
######################################################################        
my_train_data ,my_train_label =preprocessing_Number_data()
model_cnn=CNN_model()
Loop_for_computataion(my_train_data,my_train_label,model_cnn,1,1,0)
number_of_classes = 10
       


###########################################################################
###########################################################################
###########################################################################
###########################################################################

print("-------------------------Question Two------------------------------")



#part -2#The follwoing dataset is taken from#https://archive.ics.uci.edu/ml/datasets/User+Knowledge+Modeling
#
#I have slected the dataset as description mentions that its suitable to expirement for
#k-nearest neighbor algorithm
# 
#Name:User Knowledge Modeling Data Set
#Data Set Characteristics:  Multivariate 
#Number of Attributes: 5
#Number of Instances: 403
#Associated Tasks: Classification, Clustering
#
#The follwing is the measurement of the dataset:
#
#STG (The degree of study time for goal object materails), (input value)
#SCG (The degree of repetition number of user for goal object materails) (input value)
#STR (The degree of study time of user for related objects with goal object) (input value)
#LPR (The exam performance of user for related objects with goal object) (input value)
#PEG (The exam performance of user for goal objects) (input value)
#UNS (The knowledge level of user) (target value)
#

#Group of interest samples as Low and VeryLow 
#Very Low
#Low

#Group of interest samples as middle and high
#Middle
#High
#
#For the exmaple here i have taken the sample of interest as High and 
#other parameter Low, Middle, High as sample of Not interest for calculating the cohen's_d

group_array =[]
group_low =[]
group_middle =[]
group_vlow =[]
def preprocessing_userdata():
    dataset= pd.read_csv(r'D:\folder-assignment\Knn-assignment_complete\user-knowledge.csv')
    dataset.head()

    
    #grouping high and middle as one group
    gk = dataset.groupby(['UNS'])
    group_high=gk.get_group('High')
    group_high=np.array(group_high.iloc[:,])
    group_low=gk.get_group('Middle')
    group_low=np.array(group_low.iloc[:,])

    group_of_interest_highandlow=np.append(group_high,group_low,axis =0)
    group_of_interest_highandlow=pd.DataFrame(group_of_interest_highandlow)

# pritning the dimension of sample of interest
    print("Dimension of sample of interest")
    print(group_of_interest_highandlow.ndim)
    print("Size of sample of interest")
    print(len(group_of_interest_highandlow))

#grouping low and verylow as one group
    group_low=gk.get_group('Low')
    group_low=np.array(group_low.iloc[:,])
    group_vlow=gk.get_group('very_low')
    group_vlow=np.array(group_vlow.iloc[:,])
    
    group_of_interest_lowandvlow=np.append(group_low,group_vlow, axis=0)
    group_of_interest_lowandvlow=pd.DataFrame(group_of_interest_lowandvlow)

# pritning the dimension of sample of interest
    print("Dimension of sample of interest")
    print(group_of_interest_lowandvlow.ndim)
    print("Size of sample of interest")
    print(len(group_of_interest_lowandvlow))

    
    return group_of_interest_highandlow,group_of_interest_lowandvlow,dataset

sorted_cohens_d=[]
#function that is used to calculate the cohens D 
def cohens_d_calculation(column_of_interest,column_notof_interest):
    
    column_of_interest_mean=np.mean(column_of_interest)
    column_notof_interest_mean=np.mean(column_notof_interest)
    column_of_interest_std=np.std(column_of_interest)
    column_notof_interest_std=np.std(column_notof_interest)
    #older formula
    #spool=sqrt(((column_notof_interest_std**2)+(column_of_interest_std**2))/2)
    #formula taken as per suggestion from slack
    spool=sqrt(((len((column_notof_interest)-1))*(column_notof_interest_std**2)+(len((column_of_interest)-1))
    *(column_of_interest_std**2))/(len(column_notof_interest)+len(column_of_interest)-2))
    
    sorted_cohens_d.append((column_notof_interest_mean-column_of_interest_mean)/spool)
    return sorted_cohens_d

#calculating the cohends for measurement
def run_cohens_d(final_zero,final_one,dataset):
    for i in range(5):
        print("Finding the cohen's D for the measurement for sample",i)
        column_of_interest=final_one.iloc[:,i]
        column_notof_interest=final_zero.iloc[:,i]
    
        cohen_d=cohens_d_calculation(column_of_interest,column_notof_interest)
        print( dataset.columns[i]) 
        print("sorted")
        #just change the intendation so that we can  print once only  
        #https://stackoverflow.com/questions/27905351/sorting-by-absolute-value-without-changing-to-absolute-value
        print(sorted(cohen_d, key=abs))
        #print(np.sort(cohen_d))

###########################################################################
###################################QUESTION 2################################
###########################################################################
###########################################################################            
#same values of Final One and Final Zero is used        
final_zero,final_one,dataset=preprocessing_userdata()
run_cohens_d(final_zero,final_one,dataset)


error=0
score=1

###########################################################################
###########################################################################
###########################################################################
###########################################################################

#WARNING: Critical function if called in between lines will erase all the array that holds value for
#plotting and error and scores  
def clear():
    score_knn_1.clear()
    error_knn_1.clear()
    score_knn_5.clear()
    error_knn_5.clear()
    score_knn_10.clear()
    error_knn_10.clear()
    loss_cnn.clear()
    score_cnn.clear()
    error_cnn.clear()
    loss_ann.clear()
    score_ann.clear()
    error_ann.clear()
    
# Defining the Ann model for Question 3 for Individual Dataset
def model_ann_def():
    model_class = Sequential()
#    you can change the No of layers for checking error and accuracy 
    model_class.add(Dense(output_dim =18,init="uniform",activation ="relu",input_dim=5))
#    model_class.add(Dense(output_dim =18,init="uniform",activation ="relu",input_dim=5))
#    model_class.add(Dense(output_dim =18,init="uniform",activation ="relu",input_dim=5))
    model_class.add(Dense(output_dim =4,init="uniform",activation ="softmax"))
    model_class.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    return model_class


from sklearn.preprocessing import LabelEncoder
def data_preprocessing_q3():
    clear()
    dataset= pd.read_csv(r'D:\folder-assignment\Knn-assignment_complete\user-knowledge.csv')
    X = dataset.iloc[:,:-1]
    y = dataset.iloc[:,-1]
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)
    y=y.reshape(np.size(y))
    my_train_data =np.array(X)
    my_train_label =np.array(y)    
    return my_train_data,my_train_label

###########################################################################
#################################QUESTION 3################################
###########################################################################
###########################################################################
Q3_Q1=1
my_train_data,my_train_label=data_preprocessing_q3()
ann=model_ann_def()
Loop_for_computataion(my_train_data,my_train_label,1,ann,0,0)

#######################################################################
#############################QUESTION 4################################
#######################################################################


#we aree not using the sample as script is doing the same functionality
all_vars = io.loadmat(r"E:\Deepak_assignment_3_ml\NumberRecognitionTesting.mat")
test_samples= all_vars.get("X_test")
imgLent = np.prod(test_samples.shape[:-1])
lengtht = test_samples.shape[-1]
test_samples=test_samples.reshape([imgLent, lengtht])
test_samples=test_samples.T
test_samples = test_samples.reshape(len(test_samples),28,28,1)

my_train_data_u_x ,my_train_label_u_x =preprocessing_Number_data()


model = Sequential()
model.add(BatchNormalization())  
model.add(Conv2D(32,kernel_size=(3,3),input_shape=(28, 28, 1),activation='relu'))
model.add(Conv2D(64,kernel_size=(3,3),input_shape=(28, 28, 1),activation='relu'))
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Conv2D(32,kernel_size=(3,3),input_shape=(28, 28, 1),activation='relu'))

model.add(Flatten())
model.add(Dense(units=10, activation="softmax"))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=["accuracy"])


#Enable only when you want the prediction  
clear()
#syntax
#Loop_for_computataion(my_train_data, my_train_label ,model,model_ann, status,question4_enable)
Loop_for_computataion(my_train_data_u_x,my_train_label_u_x,model,0,0,1)

############################################################
#code for running the model directly without K fold  use the code below
#my_train_data_u_x = my_train_data_u_x.reshape(len(my_train_data_u_x),28,28,1)
#my_train_data_u_x = np_utils.to_categorical(my_train_label_u_x, 10)    
#model.fit(my_train_data_u_x,my_train_label_u_x,batch_size=32,epochs=15)
#predictions_csv()

