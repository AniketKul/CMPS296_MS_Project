import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
import datetime 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split 
from sklearn.cross_validation import KFold 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve
from sklearn.metrics import auc,roc_curve,roc_auc_score,classification_report
import warnings

warnings.filterwarnings('ignore')

ccdata = pd.read_csv("../dataset/creditcard.csv", header = 0)
ccdata.info()

sns.countplot("Class", data=ccdata)
plt.show()

normal_transactions_count = len(ccdata[ccdata["Class"] == 0])
fraud_transactions_count = len(ccdata[ccdata["Class"] == 1])
percentage_normal_transactions = normal_transactions_count / (normal_transactions_count + fraud_transactions_count)
percentage_fraud_transactions = fraud_transactions_count / (normal_transactions_count + fraud_transactions_count)
print("Percentage of Normal Transacations is : ", percentage_normal_transactions * 100)
print("Percentage of Fraud Transacations is : ", percentage_fraud_transactions * 100)

normal_transactions = ccdata[ccdata["Class"] == 0]
fraud_transactions = ccdata[ccdata["Class"] == 1]
plt.figure(figsize = (10, 6))
plt.subplot(121)
normal_transactions[normal_transactions["Amount"] <= 2500].Amount.plot.hist(title = "Normal Transacations")
plt.subplot(122)
fraud_transactions[fraud_transactions["Amount"] <= 2500].Amount.plot.hist(title = "Fraud Transacations")
plt.show()

# Model function for modeling with confusion matrix
def model(model, features_train, features_test, labels_train, labels_test):
    clf = model
    clf.fit(features_train, labels_train.values.ravel())
    pred = clf.predict(features_test)
    cnf_matrix = confusion_matrix(labels_test, pred)
    print("Recall:", cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))
    fig = plt.figure(figsize=(6,3))
    print("True Positives: ", cnf_matrix[1,1,]) # no of fraud transaction which are predicted fraud
    print("True Negatives: ", cnf_matrix[0,0]) # no. of normal transaction which are predited normal
    print("False Positives: ", cnf_matrix[0,1]) # no of normal transaction which are predicted fraud
    print("False Negatives: ", cnf_matrix[1,0]) # no of fraud Transaction which are predicted normal
    sns.heatmap(cnf_matrix, cmap="coolwarm_r", annot=True, linewidths = 0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    print("\n----------Classification Report------------------------------------")
    print(classification_report(labels_test,pred))


# Preparing data for training and testing since we are going to use different data.
def data_preparation(x):
    x_features = x.ix[:, x.columns != 'Class']
    x_labels = x.ix[:, x.columns == 'Class']
    x_features_train, x_features_test, x_labels_train, x_labels_test = train_test_split(x_features, x_labels, test_size = 0.3)
    print("No. of records in Training Data: ")
    print(len(x_features_train))
    print("No. of records in Test Data: ")
    print(len(x_features_test))
    return(x_features_train, x_features_test, x_labels_train, x_labels_test)


# Undersampling

# In this technique, we consider a portion of majority class and will take whole data of minority class
# count fraud transactions = the total number of fraud transactions
# now lets us see the index of normal and fraud cases.
fraud_indices= np.array(ccdata[ccdata.Class == 1].index)
normal_indices = np.array(ccdata[ccdata.Class == 0].index)

def undersample(normal_indices, fraud_indices, times): #times denote the normal data = times * fraud data
    normal_indices_undersample = np.array(np.random.choice(normal_indices, (times * fraud_transactions_count), replace=False))
    undersample_data = np.concatenate([fraud_indices, normal_indices_undersample])
    undersample_data = ccdata.iloc[undersample_data,:]
    count_normal = len(undersample_data[undersample_data.Class == 0])
    count_fraud = len(undersample_data[undersample_data.Class == 1])
    total_count = count_normal + count_fraud
    print("Percentage of Normal Transacation: ", (count_normal/total_count)*100)
    print("Percentage of Fraud Transacation: ", (count_fraud/total_count)*100)
    print("Total number of records in undersampled data: ", total_count)
    return(undersample_data)

# before starting we should standridze our ampount column
ccdata["Normalized Amount"] = StandardScaler().fit_transform(ccdata['Amount'].reshape(-1, 1))
ccdata.drop(["Time","Amount"], axis=1, inplace=True)
print(ccdata.head())

#Perform LogisticRegression with undersample data

#1st iteration conatain 50% normal transactions
#2nd iteration contains 66% noraml transactions
#3rd iteration contains 75% normal transactions


print("________________________________________________________________________________________________________")

for i in range(1, 4):
    print("Undersample data details for iteration {}".format(i))
    print()
    undersample_data = undersample(normal_indices, fraud_indices, i)
    print("------------------------------------------------------------")
    print()
    print("Logistic Regression classification for iteration {}".format(i))
    print()
    undersample_features_train, undersample_features_test, undersample_labels_train, undersample_labels_test = data_preparation(undersample_data)
    #the partion for whole data
    print()
    clf = LogisticRegression()
    model(clf, undersample_features_train, undersample_features_test, undersample_labels_train, undersample_labels_test)
    print("________________________________________________________________________________________________________")


print()
print("*** LogisticRegression - Training the model using undersample data and testing the whole test dataset ****")
print()

#training the model using undersample data and testing it for the whole data test set
for i in range(1, 4):
    print("Undersample data details for iteration {}".format(i))
    print()
    undersample_data = undersample(normal_indices, fraud_indices, i)
    print("------------------------------------------------------------")
    print()
    print("Logistic Regression for iteration {}".format(i))
    print()
    undersample_features_train, undersample_features_test, undersample_labels_train, undersample_labels_test = data_preparation(undersample_data)

    #for whole dataset
    data_features_train, data_features_test, data_labels_train, data_labels_test = data_preparation(ccdata)
    print()
    clf = LogisticRegression()
    #training with the undersampled data and testing with whole data
    model(clf, undersample_features_train, data_features_test, undersample_labels_train, data_labels_test)
    
    print("_________________________________________________________________________________________")


print()
print("*** Support Vector Machine with undersample Data ***")
#Using Support Vector Machine with undersampled Data
for i in range(1,4):
    print("the undersample data for iteration {}".format(i))
    print()
    undersample_data = undersample(normal_indices,fraud_indices,i)
    print("------------------------------------------------------------")
    print()
    print("Support Vector Machine for iteration {}".format(i))
    print()
    undersample_features_train, undersample_features_test, undersample_labels_train, undersample_labels_test = data_preparation(undersample_data)
    print()
    clf = SVC()
    model(clf, undersample_features_train, undersample_features_test, undersample_labels_train, undersample_labels_test)
    print("________________________________________________________________________________________________________")


print()
print("*** SVM - Training the model using undersample data and testing the whole test dataset ***")
print()
#SVM -  training this model using undersample data and test for the whole data test set
for i in range(1,4):
    print("the undersample data for iteration {}".format(i))
    print()
    undersample_data = undersample(normal_indices,fraud_indices,i)
    print("------------------------------------------------------------")
    print()
    print("Support Vector Machine for iteration {}".format(i))
    print()
    undersample_features_train, undersample_features_test, undersample_labels_train, undersample_labels_test = data_preparation(undersample_data)
    data_features_train,data_features_test,data_labels_train,data_labels_test = data_preparation(ccdata)

    print()
    clf = SVC()
    model(clf, undersample_features_train, data_features_test, undersample_labels_train, data_labels_test)
    print("_________________________________________________________________________________________")


print()
print("*** RandomForest with undersample Data ***")
#RandomForest with undersampled data
for i in range(1,4):
    print("the undersample data for iteration {}".format(i))
    print()
    undersample_data = undersample(normal_indices,fraud_indices,i)
    print("------------------------------------------------------------")
    print()
    print("RandomForest for iteration {}".format(i))
    print()
    undersample_features_train, undersample_features_test, undersample_labels_train, undersample_labels_test = data_preparation(undersample_data)
    print()
    clf = RandomForestClassifier(n_estimators=100)
    model(clf, undersample_features_train, undersample_features_test, undersample_labels_train, undersample_labels_test)
    print("________________________________________________________________________________________________________")

print()
print("*** RandomForest - Training the model using undersample data and testing the whole test dataset ***")
print()
#RandomForest - training this model using undersample data and test for the whole test dataset
for i in range(1,4):
    print("the undersample data for iteration {}".format(i))
    print()
    undersample_data = undersample(normal_indices,fraud_indices,i)
    print("------------------------------------------------------------")
    print()
    print("RandomForest for iteration {}".format(i))
    print()
    undersample_features_train, undersample_features_test, undersample_labels_train, undersample_labels_test = data_preparation(undersample_data)
    data_features_train,data_features_test,data_labels_train,data_labels_test = data_preparation(ccdata)
    print()
    clf = RandomForestClassifier(n_estimators=100)
    model(clf, undersample_features_train, data_features_test, undersample_labels_train, data_labels_test)
    print("_________________________________________________________________________________________")


#Oversampling
ccdata = pd.read_csv("../dataset/creditcard.csv", header = 0)
print("Length of training data: ", len(ccdata))
print("Length of Normal data: ",len(ccdata[ccdata["Class"]==0]))
print("Length of Fraud data: ",len(ccdata[ccdata["Class"]==1]))

data_train_X, data_test_X, data_train_y, data_test_y = data_preparation(ccdata)
data_train_X.columns
data_train_y.columns

# We have a training data
data_train_X["Class"] = data_train_y["Class"] 
data_train = data_train_X.copy() 
print("length of training data",len(data_train))

normal_data = data_train[data_train["Class"]==0]
print("length of normal data",len(normal_data))
fraud_data = data_train[data_train["Class"]==1]
print("length of fraud data",len(fraud_data))


for i in range (300): 
    normal_data = normal_data.append(fraud_data)
os_data = normal_data.copy()
print("Length of Oversampled data: ", len(os_data))
print("Number of normal transcations in Oversampled data: ", len(os_data[os_data["Class"]==0]))
print("No. of fraud transcations: ", len(os_data[os_data["Class"]==1]))
print("Proportion of Normal data in Oversampled data is: ", len(os_data[os_data["Class"]==0])/len(os_data))
print("Proportion of Fraud data in Oversampled data is: ", len(os_data[os_data["Class"]==1])/len(os_data))

os_data["Normalized Amount"] = StandardScaler().fit_transform(os_data['Amount'].reshape(-1, 1))
os_data.drop(["Time", "Amount"], axis=1, inplace = True)
os_data.head()


os_train_X, os_test_X, os_train_y, os_test_y = data_preparation(os_data)
#clf = RandomForestClassifier(n_estimators=100)
print("$$ LogisticRegression $$")
clf = LogisticRegression()
model(clf, os_train_X, os_test_X, os_train_y, os_test_y)
print("$$ SVM $$")
# clf = SVC()
# model(clf, os_train_X, os_test_X, os_train_y, os_test_y)
print("$$ Random Forest $$")
clf = RandomForestClassifier(n_estimators=100)
model(clf, os_train_X, os_test_X, os_train_y, os_test_y)


# now take all over sampled data as training and test it for test data
os_data_X = os_data.ix[:,os_data.columns != "Class"]
os_data_y = os_data.ix[:,os_data.columns == "Class"]

data_test_X["Normalized Amount"] = StandardScaler().fit_transform(data_test_X['Amount'].reshape(-1, 1))
data_test_X.drop(["Time","Amount"],axis=1,inplace=True)
data_test_X.head()

#clf= RandomForestClassifier(n_estimators = 100)
#clf = LogisticRegression()
print("$$ LogisticRegression $$")
clf = LogisticRegression()
model(clf, os_data_X, data_test_X, os_data_y, data_test_y)
print("$$ SVM $$")
clf = SVC()
model(clf, os_data_X, data_test_X, os_data_y, data_test_y)
print("$$ Random Forest $$")
clf = RandomForestClassifier(n_estimators=100)
model(clf, os_data_X, data_test_X, os_data_y, data_test_y)


print()
print("Using SMOTE for Oversampling")
# Lets Use SMOTE for Sampling
from imblearn.over_sampling import SMOTE
data = pd.read_csv('../dataset/creditcard.csv')
os = SMOTE(random_state = 0) 

data_train_X, data_test_X, data_train_y, data_test_y = data_preparation(data)
columns = data_train_X.columns

os_data_X, os_data_y = os.fit_sample(data_train_X, data_train_y)
os_data_X = pd.DataFrame(data = os_data_X, columns = columns)
os_data_y= pd.DataFrame(data = os_data_y, columns = ["Class"])

print("length of oversampled data is ",len(os_data_X))
print("Number of normal transcation in oversampled data",len(os_data_y[os_data_y["Class"]==0]))
print("No.of fraud transcation",len(os_data_y[os_data_y["Class"]==1]))
print("Proportion of Normal data in oversampled data is ",len(os_data_y[os_data_y["Class"]==0])/len(os_data_X))
print("Proportion of fraud data in oversampled data is ",len(os_data_y[os_data_y["Class"]==1])/len(os_data_X))

os_data_X["Normalized Amount"] = StandardScaler().fit_transform(os_data_X['Amount'].reshape(-1, 1))
os_data_X.drop(["Time","Amount"], axis = 1, inplace = True)
data_test_X["Normalized Amount"] = StandardScaler().fit_transform(data_test_X['Amount'].reshape(-1, 1))
data_test_X.drop(["Time","Amount"], axis = 1, inplace = True)

print("SMOTE - LR: ")
clf = LogisticRegression()
# training the data using oversampled data and predicting frauds for the test data.
model(clf, os_data_X, data_test_X, os_data_y, data_test_y)

print("SMOTE - SVM: ")
clf = SVC()
model(clf, os_data_X, data_test_X, os_data_y, data_test_y)

print("SMOTE - RandomForest: ")
clf = RandomForestClassifier(n_estimators = 100)
model(clf, os_data_X, data_test_X, os_data_y, data_test_y)
