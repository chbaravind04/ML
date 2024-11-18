import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
##############################From Tabular data##############################################33
# Define the data
data = {
    'CGPA': ['g9', 'g8', 'g9', 'l8', 'g8', 'g9', 'l8', 'g9', 'g8', 'g8'],
    'Inter': ['Y', 'N', 'N', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'Y'],
    'PK': ['+++', '+', '==', '==', '+', '+', '+', '+++', '+', '=='],
    'CS': ['G', 'M', 'P', 'G', 'M', 'M', 'P', 'G', 'G', 'G'],
    'Job': ['Y', 'Y', 'N', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'Y']
}

# Create DataFrame
table = pd.DataFrame(data)

# Convert categorical variables to numerical using LabelEncoder
encoder = LabelEncoder()
for i in table.columns:
    table[i] = encoder.fit_transform(table[i])

# Split the data into features and target
X = table.iloc[:, 0:4]
y = table.iloc[:, 4]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

# Create and train the Decision Tree model use CART and ID3 Methods- 'entropy' and 'gini'
model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate and print the accuracy of the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
################################FOR IRIS DATA SET############################################
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Train the model using DecisionTreeClassifier ID3
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model = clf.fit(X, y)
fig = plt.figure(figsize=(10,8))
_ = tree.plot_tree(clf,
feature_names=iris.feature_names,
class_names=iris.target_names,
filled=True)
plt.show()
#fig.savefig("decistion_tree.png")
