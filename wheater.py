import pandas as pd #for data manipulation
import numpy as np #for converting data into array
from sklearn.model_selection import train_test_split #for training model & sepration of data & training set
from sklearn.ensemble import RandomForestRegressor #importing the model we are using
import random
#Reason for choosing this:
#Random forest algorithm can be used for both classifications and regression task.
#It provides higher accuracy.
#Random forest classifier will handle the missing values and maintain the accuracy of a large proportion of data.
#If there are more trees, it won’t allow overfitting trees in the model.
#It has the power to handle a large data set with higher dimensionality


features = pd.read_csv('/Users/abcd/Desktop/Weather-prediction/temps.csv')
print(features.head(5)) #feature because they are IP and in ML inputs are known as features and head will select 5 cols from dataset

print('The shape of our features is:', features.shape) #shape method tells us the matrix dimension we have in our dataset

print(features.describe()) #describe for Descriptive statistics for each column

features = pd.get_dummies(features) #dummies convert catgeroial variable to dummy variable

print(features.iloc[:,5:].head(5)) # Display the first 5 rows of the last 12 columns,iloc is used for position based selection in pandas.

labels = np.array(features['actual']) #labels will the actual value we want to predict

#removing labels from features
features= features.drop('actual', axis = 1) #axis  1 refers to the columuns

feature_list = list(features.columns) #saving features name

features = np.array(features)
#print(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42) # Split the data into training and testing sets
#test_size represent the proportion of the dataset to include in the test split.
#random_state is the seed used by the random number generator

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

baseline_preds = test_features[:, feature_list.index('average')] #historical average preds
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Training the model on training data
rf.fit(train_features, train_labels)

#print(test_features[0])
#print(test_features[0][0],test_features[0][1],test_features[0][2],test_features[0][3],test_features[0][4],test_features[0][5],test_features[0][6],test_features[0][7],test_features[0][8],test_features[0][9],test_features[0][10],test_features[0][11],test_features[0][12],test_features[0][13],test_features[0][14],test_features[0][15],test_features[0][16])
temp1 = random.randint(1,60)
temp2 = random.randint(1,70)
actual = random.randint(1,60)
test = [2019.0,10.0,28.0,temp1,temp2,actual,65.0,74.0,53.0,45.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]
print(test)
# Use the forest's predict method on the test data
predictions = rf.predict([test])

print('Predicted Temperature:',predictions[0])

# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')