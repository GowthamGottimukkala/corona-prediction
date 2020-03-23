#### Explanation - Please read this before proceeding ####
# Initially I converted the outputvariable in the training set from probability to 0&1 using a threshold of 50%
# In the preprocessing step I used Knearest Neighbours to fill in the missing values instead of using mean and mode
# Also I didn't standardize the data because Gradient boosting algorithm doesn't assume the data to be in guassian distribution unlike knn algorithm.
# I used Gradient boosting algorithm for training the classifier on 70% of training data and tested on the remaining 30% to get an accuracy of 90.4 percent(accuracy_score)
# Now I trained on the 100% training data to predict probabilites for the given test data and exported them to output.csv

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.impute import KNNImputer

# Input
pd.options.mode.chained_assignment = None
df = pd.read_excel("Train_dataset.xlsx", sheet_name="Train_dataset")
formarch20 = pd.read_excel("Train_dataset.xlsx", sheet_name="Diuresis_TS")
df_test = pd.read_excel("Test_dataset.xlsx", sheet_name="Test_dataset")
y_target = df['Infect_Prob']
y_target = preprocessing.Binarizer(50).fit_transform(pd.DataFrame(y_target))
x_train = df[df.columns.difference(['people_ID','Designation','Name','Insurance','Infect_Prob'])]
x_train["Diuresis"] = formarch20.iloc[:,1]
x_test = df_test[df_test.columns.difference(['people_ID','Designation','Name','Insurance'])]
print(x_train.isnull().sum())
print(x_train.head())
print(x_test.head())

def handle_non_numerical_data(df):
    global x_train, x_test
    x_train = pd.get_dummies(x_train, columns=["Gender", "Region","Married",
         "Occupation", "Mode_transport","comorbidity", "Pulmonary score",
          "cardiological pressure",], prefix=["Gender", "region","married","occupation",
           "mode_transport","comorbidity","pulmonary score","cardiological pressure"])
    x_test = pd.get_dummies(x_test, columns=["Gender", "Region","Married",
      "Occupation", "Mode_transport","comorbidity", "Pulmonary score",
       "cardiological pressure",], prefix=["Gender", "region","married","occupation",
        "mode_transport","comorbidity","pulmonary score","cardiological pressure"])

def pre(z):
    columns = z.columns.values
    for column in columns:
        min_max_scaler = preprocessing.MinMaxScaler()
        z[column] = min_max_scaler.fit_transform(pd.DataFrame(z[column]))
    return z

# There are some regions in test that train never saw.. so added them in train and initialized with 0's
unique_test = [x for x in x_test.Region.unique() if x not in x_train.Region.unique()]
unique_train = [x for x in x_train.Region.unique() if x not in x_test.Region.unique()]
for elem in unique_train:
    x_test["region_"+elem] = 0
for elem in unique_test:
    x_train["region_"+elem] = 0
print(x_train)
print(x_test)

# Changing categorical to Numerical
handle_non_numerical_data(x_train)
x_train = x_train.reindex(sorted(x_train.columns), axis=1)
x_test = x_test.reindex(sorted(x_test.columns), axis=1)
print(x_train.head())
print(x_test.head())

# Filling Null Values using KNN (instead of mean and mode)
cols = x_train.columns
imputer = KNNImputer(n_neighbors=5)
x_train = pd.DataFrame(imputer.fit_transform(x_train))
x_train.columns = cols
print(x_train.head())

# Rescaling values - Preprocessing
x_train = pre(x_train)
x_test = pre(x_test)
print(x_train.head())
print(x_test.head())


# Model Starts (for entire training set)
clf = GradientBoostingClassifier(n_estimators=200)
clf.fit(x_train,y_target)
y_test = clf.predict_proba(x_test)
print(y_test)
probofone = pd.DataFrame(y_test).iloc[:,1]
answer = pd.DataFrame(df_test.iloc[:,0])
answer["Infect_Prob"] = probofone

# Exporting the answer
export_csv = answer.to_csv (r"output.csv", index = None, header=True)


# ############### for the training  ######################
# # Model Starts (for split)
# print(x_train.shape,y_target.shape)
# X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_target, test_size = 0.4, random_state = 100)
# clf = GradientBoostingClassifier(n_estimators=200)
# clf.fit(X_train, Y_train)
# Y_pred = clf.predict(X_test)
# predictions = clf.predict_proba(X_test) # these are the probabilites
# results = confusion_matrix(Y_test, Y_pred) 
# print("ConfusionMatrix\n",results)
# print("Accuracy Score - ",accuracy_score(Y_test,Y_pred))
# ############### I got 90.4% accuracy ######################