# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 08:13:55 2024

@author: zone
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 08:08:45 2024

@author: zone
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix

CGPA = ['g9', 'g8', 'g9', 'l8', 'g8', 'g9', 'l8', 'g9', 'g8', 'g8']
Inter = ['Y', 'N', 'N', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'Y']
PK = ['+++', '+', '==', '==', '+', '+', '+', '+++', '+', '==']
CS = ['G', 'M', 'P', 'G', 'M', 'M', 'P', 'G', 'G', 'G']
Job = ['Y', 'Y', 'N', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'Y']

# Creating LabelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers
CGPA_encoded = le.fit_transform(CGPA)
Inter_encoded = le.fit_transform(Inter)
PK_encoded = le.fit_transform(PK)
CS_encoded = le.fit_transform(CS)
label = le.fit_transform(Job)

# Print encoded labels
print("CGPA:", CGPA_encoded)
print("Inter:", Inter_encoded)
print("PK:", PK_encoded)
print("CS:", CS_encoded)
print("Job:", label)

# Prepare the features set
features = []
for i in range(len(CGPA_encoded)):
    features.append([CGPA_encoded[i], Inter_encoded[i], PK_encoded[i], CS_encoded[i]])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.30, random_state=2)

# Create a Gaussian Naive Bayes classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(X_train, y_train)

# Predict the output for the test set
y_pred = model.predict(X_test)

# Print classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Predict if a new example gets the job or not
example = [[2, 0, 2, 0]]  # New example for prediction
prediction = model.predict(example)

# Print the prediction result
if prediction == 1:
    print("Predicted Value: Got JOB")
else:
    print("Predicted Value: Didn't get JOB")
