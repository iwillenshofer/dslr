# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
# helps to make it converg better
from sklearn.preprocessing import StandardScaler


FEATURES = [
    'Arithmancy',
    'Astronomy',
    'Herbology',
    'Defense Against the Dark Arts',
    'Divination',
    'Muggle Studies',
    'Ancient Runes',
    'History of Magic',
    'Transfiguration',
    'Potions',
    'Care of Magical Creatures',
    'Charms',
    'Flying'
]
# Load your dataset
data = pd.read_csv('./datasets/dataset_train.csv')

# Encode the 'Best Hand' column as binary (0 for Left, 1 for Right)
data['Best Hand'] = LabelEncoder().fit_transform(data['Best Hand'])

# Split the data into training and testing sets
X = data[FEATURES]
y = data['Hogwarts House']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# remove lines with missing data.
X_train.dropna(inplace=True)
y_train = y_train[X_train.index]  # Update y_train accordingly

X_test.dropna(inplace=True)
# print(y_test.shape)
y_test = y_test[X_test.index]  # Update y_test accordingly
# print(y_test.shape)


# help achive better convergense 
#  Rescaling the input features can sometimes help with convergence. 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a logistic regression model
model = LogisticRegression()

# modofy the number of itaration
# model = LogisticRegression(max_iter=1000)  # Increase max_iter to a higher value


# Train the model on the training data
# model.fit(X_train, y_train)


# use rescaled data
model.fit(X_train_scaled, y_train)

# Make predictions on the test data
# y_pred = model.predict(X_test)

# rescaled
y_pred = model.predict(X_test_scaled )

print(X_test_scaled)
print(y_pred)

# Calculate accuracy and display the confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)


import seaborn as sns
import matplotlib.pyplot as plt

# Create a confusion matrix heatmap
# Confusion Matrix Heatmap: Visualize the confusion matrix as a heatmap 
# to see how well your model is classifying instances.
# confusion = confusion_matrix(y_test, y_pred)
# sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()


#  bar chart showing the coefficients of the logistic regression model
# Get feature coefficients
feature_names = X.columns
coefficients = model.coef_[0]

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefficients)
plt.xlabel('Coefficient Value')
plt.title('Feature Importance')
plt.show()


# # Receiver Operating Characteristic (ROC) Curve:
# # The ROC curve is a graphical representation of a binary classifier's performance.

# from sklearn.metrics import roc_curve, roc_auc_score

# # Get predicted probabilities
# y_pred_prob = model.predict_proba(X_test_scaled)[:,1]

# # Calculate ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# # Plot ROC curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, linewidth=2)
# plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.show()
