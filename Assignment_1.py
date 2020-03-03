#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:45:50 2019

@author: poojatyagi
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score
from scipy.stats import zscore



col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","complex"]

train_data = pd.read_csv('/Users/poojatyagi/Desktop/Fraud and anomaly/NSL-KDD/KDDTrain+.txt', header= None, names = col_names)
test_data = pd.read_csv('/Users/poojatyagi/Desktop/Fraud and anomaly/NSL-KDD/KDDTest+.txt', header= None, names = col_names)


train_data.dtypes
test_data.shape
##########################################################################################################
#Pre-processing dataset
#removing complex column as our target feature is types of attack from traina nd test data
X_train = train_data.drop(['complex'], axis=1) 
X_test = test_data.drop(['complex'], axis=1)

# DoS,Probe, U2R, R2L attackes as category abnormal... 0- normal and 1 - abnormal -- TRAIN DATA
X_train.loc[(X_train.label == 'back')
 | (X_train.label == 'land') 
 | (X_train.label == 'neptune')
 | (X_train.label == 'pod')
 | (X_train.label == 'smurf') 
 | (X_train.label == 'teardrop') 
 | (X_train.label == 'apache2')
 | (X_train.label == 'udpstorm')
 | (X_train.label == 'processtable') 
 | (X_train.label == 'worm')
 | (X_train.label == 'satan') 
 | (X_train.label == 'ipsweep') 
 | (X_train.label == 'nmap')
 | (X_train.label == 'portsweep')
 | (X_train.label == 'mscan')
 | (X_train.label == 'saint')
 | (X_train.label == 'ftp_write') 
 | (X_train.label == 'imap')
 | (X_train.label == 'phf')
 | (X_train.label == 'multihop') 
 | (X_train.label == 'warezmaster') 
 | (X_train.label == 'warezclient ') 
 | (X_train.label == 'spy')
 | (X_train.label == 'xlock') 
 | (X_train.label == 'xsnoop') 
 | (X_train.label == 'snmpguess')
 | (X_train.label == 'snmpgetattack') 
 | (X_train.label == 'httptunnel')
 | (X_train.label == 'sendmail')
 | (X_train.label == 'named')
 | (X_train.label == 'buffer_overflow')
 | (X_train.label == 'loadmodule') 
 | (X_train.label == 'rootkit')
 | (X_train.label == 'perl') 
 | (X_train.label == 'sqlattack') 
 | (X_train.label == 'xterm')
 | (X_train.label == 'ps'), 'class']= 1    

#NORMAL
X_train.loc[(X_train.label == 'normal'), 'class']= 0

#DoS,Probe, U2R, R2L attackes as category abnormal... 0- normal and 1 - abnormal-- TEST DATA
    
#DoS 
X_test.loc[(X_test.label == 'back')
 | (X_test.label == 'land') 
 | (X_test.label == 'neptune')
 | (X_test.label == 'pod')
 | (X_test.label == 'smurf') 
 | (X_test.label == 'teardrop') 
 | (X_test.label == 'apache2')
 | (X_test.label == 'udpstorm')
 | (X_test.label == 'processtable') 
 | (X_test.label == 'worm')
 | (X_test.label == 'satan') 
 | (X_test.label == 'ipsweep') 
 | (X_test.label == 'nmap')
 | (X_test.label == 'portsweep')
 | (X_test.label == 'mscan')
 | (X_test.label == 'saint')
 | (X_test.label == 'ftp_write') 
 | (X_test.label == 'imap')
 | (X_test.label == 'phf')
 | (X_test.label == 'multihop') 
 | (X_test.label == 'warezmaster') 
 | (X_test.label == 'warezclient ') 
 | (X_test.label == 'spy')
 | (X_test.label == 'xlock') 
 | (X_test.label == 'xsnoop') 
 | (X_test.label == 'snmpguess')
 | (X_test.label == 'snmpgetattack') 
 | (X_test.label == 'httptunnel')
 | (X_test.label == 'sendmail')
 | (X_test.label == 'named')
 | (X_test.label == 'buffer_overflow')
 | (X_test.label == 'loadmodule') 
 | (X_test.label == 'rootkit')
 | (X_test.label == 'perl') 
 | (X_test.label == 'sqlattack') 
 | (X_test.label == 'xterm')
 | (X_test.label == 'ps'), 'class']= 1   

#NORMAL
X_test.loc[(X_test.label == 'normal'), 'class']= 0

###############################################################################################################
#splitting input feature and target feature of TRAIN data
X_train['class'].isna().sum() #checking if any null values
X_train = X_train.dropna() #dropping instances that have null values
train_data_X = X_train.iloc[:,0:len(X_train.columns)-2] #input features
train_data_Y = X_train.loc[:,'class']
train_data_Y = train_data_Y.astype('int')  #target feature 'class'

#splitting input feature and target feature of TEST data

X_test['class'].isna().sum() #checking if any null values
X_test = X_test.dropna() #dropping instances that have null values
test_data_X = X_test.iloc[:,0:len(X_test.columns)-2] #input features
test_data_Y = X_test.loc[:,'class']
test_data_Y = test_data_Y.astype('int')  #target feature 'class'

###############################################################################################################
#ONE-HOT ENCODING for 3 nominal features ['protocol_type','flag','service']
le = preprocessing.LabelEncoder()
enc = OneHotEncoder()
lb = preprocessing.LabelBinarizer()

train_data_X['protocol_type'] = le.fit_transform(train_data_X['protocol_type'])
test_data_X['protocol_type'] = le.fit_transform(test_data_X['protocol_type'])

train_data_X['flag'] = le.fit_transform(train_data_X['flag'])
test_data_X['flag'] = le.fit_transform(test_data_X['flag'])

train_data_X['service'] = le.fit_transform(train_data_X['service'])
test_data_X['service'] = le.fit_transform(test_data_X['service'])


########################################################################################################
#DATA VISUALIZATION

# Calculating standard devistion of features excluding continous features
std_list = ['protocol_type', 'service', 'flag','root_shell', 'land', 'logged_in', 'su_attempted', 'is_host_login', 'is_guest_login']
std_train = train_data_X.drop(std_list, axis=1)

#drop n smallest std features
stdtrain = std_train.std(axis=0)
std_X_train = stdtrain.to_frame()
std_X_train.nsmallest(10, columns=0).head(10)

#num_outbound_cmds has 0 std so will drop that feature
train_data_X = train_data_X.drop(['num_outbound_cmds'], axis=1)
test_data_X = test_data_X.drop(['num_outbound_cmds'], axis=1)

# Making list of 10 features with lowest standard deviation
stdrop_list = ['urgent', 'num_shells', 'root_shell',
        'num_failed_logins', 'num_access_files', 'dst_host_srv_diff_host_rate',
        'diff_srv_rate', 'dst_host_diff_srv_rate', 'wrong_fragment']

X_test_stdrop = test_data_X.drop(stdrop_list, axis=1)
X_train_stdrop = train_data_X.drop(stdrop_list, axis=1)
df_train_stdrop = pd.concat([X_train_stdrop, train_data_Y], axis=1)

#######################################################################################################
#OUTLIERS DETECTION
#Feature 'Duration'

traiNData_OD = train_data_X.copy()
traiNData_OD["duration_zscore"] = zscore(traiNData_OD["duration"])
traiNData_OD["is_outlier"] = traiNData_OD["duration_zscore"].apply(lambda x: x <= -3 or x >= 3)

traiNData_OD[traiNData_OD["is_outlier"]]


########################################################################################################
#FEATURE SELECTION
#Ensemble feature selection (Random forst classifier and Gradient Boosting classifier)
RF = RandomForestClassifier(n_estimators=10, criterion='entropy', max_features='auto', bootstrap=True)
GB = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200, max_features='auto')
ET = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_features='auto', bootstrap=False)

y_train = train_data_Y.loc[:].ravel()
x_train = train_data_X.values
x_test = test_data_X.values


RF.fit(train_data_X, train_data_Y)
RF_feature = RF.feature_importances_
RF_feature
rf_score = RF.score(test_data_X, test_data_Y)
print("RandomForestClassifier score is:", rf_score)


GB.fit(train_data_X, train_data_Y)
GB_feature = GB.feature_importances_
GB_feature
gb_score = GB.score(test_data_X, test_data_Y)
print("GradientBoostingClassifier score is:", gb_score)

ET.fit(train_data_X, train_data_Y)
ET_feature = ET.feature_importances_
ET_feature
et_score = ET.score(test_data_X, test_data_Y)
print("ExtraTreeClassifier score is:", et_score)

#Representing how features affect each other via ensembling
cols = train_data_X.columns.values
features_effect = pd.DataFrame({'features': cols,
                           'RandomForest' : RF_feature,
                           'ExtraTree' : ET_feature,
                           'GradientBoost' : GB_feature
                          })


#Plotting graph showing the individual features impact
graph = features_effect.plot.bar(figsize = (18, 10), title = 'Feature distribution', grid=True, legend=True, fontsize = 15, 
                            xticks=features_effect.index)
graph.set_xticklabels(features_effect.features, rotation = 80)    



#Now selecting the top 15 features from all the ensembling outputs
random_feat = features_effect.nlargest(15, 'RandomForest')
grad_feat = features_effect.nlargest(15, 'GradientBoost')
extra_feat = features_effect.nlargest(15, 'ExtraTree')

#removing the duplicates features if any
final_features = pd.concat([extra_feat, grad_feat, random_feat])
final_features = final_features.drop_duplicates() # delete duplicate feature
selected_features = final_features['features'].values.tolist()

#Preparing the dataset that includes only selected features
X_train_ens = train_data_X[selected_features]
X_test_ens = test_data_X[selected_features]



##################################################### APPLYING ALGORITHMS ON ENSEMBLED FEATURED DATASET #################################
# [1] Applying DECISION TREE classification
Deci_tree = DecisionTreeClassifier()
Deci_tree.fit(X_train_ens,train_data_Y)
pred_Deci_tree = Deci_tree.predict(X_test_ens)

# [2] RANDOM FOREST CLASSIDFIER
RF.fit(X_train_ens, train_data_Y)
pred_Random_forest = RF.predict(X_test_ens)

# [3] Gradient Boosting classifier
GB.fit(X_train_ens, train_data_Y)
pred_Gradient_boost = GB.predict(X_test_ens)

# [4] Logistic Regression
model = LogisticRegression()
model.fit(X_train_ens, train_data_Y)
predicted = model.predict(X_test_ens)
matrix = confusion_matrix(test_data_Y, predicted)

#Calculating accuracy score of all models
conclusion = pd.DataFrame({'models': ["LOGISTIC REGRESSION","DECISION TREE CLASSIFIER","RANDOMFOREST CLASSIFIER","GRADIENT BOOSTING CLASSIFIER"],
                           'accuracies': [accuracy_score(test_data_Y, predicted),accuracy_score(test_data_Y,pred_Deci_tree),
                                          accuracy_score(test_data_Y, pred_Random_forest),accuracy_score(test_data_Y,pred_Gradient_boost)]})
#### Confusion_matrix_metric
matrix1 = confusion_matrix(test_data_Y, predicted)
matrix2 = confusion_matrix(test_data_Y,pred_Deci_tree)
matrix3 = confusion_matrix(test_data_Y, pred_Random_forest)
matrix4 = confusion_matrix(test_data_Y,pred_Gradient_boost)

###############################################################################################################3
#CROSS-VALIDATION SCORE

""" 2. PERFORMING CROSS VALIDATION FOR:
 1.RANDOM FOREST
 2.LOGISTIC REGRESSION 
 3.DECISION TREE 
 4.GRADIENT BOOSTING CLASSIFIER"""
 
#1. Random forest
rfc_eval= cross_val_score(estimator = RF, X = X_train_ens, y = train_data_Y, cv = 10)
rfc_eval.mean()

#2. logistic regression
logic_reg_eval= cross_val_score(estimator = model, X = X_train_ens, y = train_data_Y, cv = 10)
logic_reg_eval.mean()

#3. Decision Tree
Deci_tree_eval= cross_val_score(estimator = Deci_tree, X = X_train_ens, y = train_data_Y, cv = 10)
Deci_tree_eval.mean()

#4. SGDC
gb_eval= cross_val_score(estimator = GB, X = X_train_ens, y = train_data_Y, cv = 10)
gb_eval.mean()

#DISPLAYING THE IMPROVED TABLE OF ACCURACY OF ALL MODELS AFTER HYPER PARAMETER OPTIMIZATION
Summary_table = pd.DataFrame([logic_reg_eval.mean(),Deci_tree_eval.mean(),rfc_eval.mean(),gb_eval.mean()],
                              index=['Logistic Regression', 'Decision Tree', 'Random Forest', 'GRDIENT BOOSTING'], columns=['Accuracy Score'])


########################################### PRINTING PART###################################################################
print("Accuracy score:")
print(conclusion)

print("Confusion matrix score: \n")
print("1. Decision Tree \n", matrix2,"\n")
print("2. Random Forest \n", matrix3,"\n")
print("3. Decision Tree \n", matrix4,"\n")

print("Summary Table for Section 1.2")
Summary_table