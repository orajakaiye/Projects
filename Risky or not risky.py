#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting

# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


# generate regression data set
X, y = make_classification(n_samples=400, n_features=1, 
	n_redundant=0, n_informative = 1, n_classes=2, 
	n_clusters_per_class=1, flip_y=0.4, random_state=0)


# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
	test_size= 1/3, random_state=0)

# construct model and fit to the training data
logre = LogisticRegression()
logre.fit(X_train, y_train)


# output the accuracy score
print('Our Accuracy is %.2f' % logre.score(X_test, y_test))

# output the number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
		% (X_test.shape[0], (y_test != logre.predict(X_test)).sum()))


# visualise the model
fig1, ax1 = plt.subplots()

ax1.scatter(X_test, y_test, color='blue')
ax1.scatter(X_test, logre.predict(X_test), color='red', marker='*')
ax1.scatter(X_test, logre.predict_proba(X_test)[:,1], color='green', marker='.')


ax1.set_xlabel('X')
ax1.set_ylabel('y')

fig1.savefig('Class_plot.png')


# In[12]:


import pandas as pd  # For data processing
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the datasets (make sure the CSV files are in the same folder)
customer_data = pd.read_csv('customer_data.csv')
payment_data = pd.read_csv('payment_data.csv')

# Merge the datasets on the 'id' column
merged_data = pd.merge(customer_data, payment_data, on='id')

# Preprocess the data
# Drop unnecessary columns and fill missing values
merged_data = merged_data.drop(['id', 'update_date', 'report_date'], axis=1, errors='ignore')
merged_data.fillna(merged_data.mean(), inplace=True)  # Fill missing values with column means

# Convert categorical variables to numeric if necessary (e.g., encoding gender)
if 'Gender' in merged_data.columns:
    merged_data['Gender'] = merged_data['Gender'].map({'Male': 1, 'Female': 0})

# Define features (X) and target variable (y)
X = merged_data.drop('label', axis=1).values  # Features (all except 'label')
y = merged_data['label'].values  # Target variable (0: Not Risky, 1: Risky)

# Normalize the features
X = StandardScaler().fit_transform(X)

# Split the data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train the logistic regression model
logre = LogisticRegression()
logre.fit(X_train, y_train)

# Generate predicted probabilities for the test set
predicted_probabilities = logre.predict_proba(X_test)[:, 1]  # Probability of being 'risky'

# Create a scatter plot with an S-shape appearance
plt.figure(figsize=(8, 5))
# We use the first feature for the X-axis; adjust the indexing as needed based on your dataset.
plt.scatter(X_test[:, 0], predicted_probabilities, c=y_test, cmap='bwr', alpha=0.7, edgecolors='k')
plt.title('S-Shaped Scatter Plot of Predicted Risk Probabilities')
plt.xlabel('Feature 1 Values')  # Adjust according to your feature names
plt.ylabel('Predicted Probability of Being Risky')
plt.axhline(0.5, color='grey', linestyle='--', label='Threshold (0.5)')  # Threshold line for reference
plt.colorbar(label='True Label (0: Not Risky, 1: Risky)')  # Color bar for clarity
plt.legend()
plt.grid(True)  # Add grid for better readability
plt.show()


# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load customer_data and payment_data
customer_data = pd.read_csv('customer_data.csv')
payment_data = pd.read_csv('payment_data.csv')

# Merge datasets on 'id'
merged_data = pd.merge(customer_data, payment_data, on='id')

# Drop irrelevant columns
merged_data = merged_data.drop(['id', 'update_date', 'report_date'], axis=1)

# Fill missing values
merged_data.fillna(merged_data.mean(), inplace=True)

# Separate features and target
X = merged_data.drop('label', axis=1).values  # All features except the target label
y = merged_data['label'].values  # The target (risky or not)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=67)

# Train the logistic regression model
logre = LogisticRegression()
logre.fit(X_train, y_train)

# Print accuracy score
print(f'Accuracy: {logre.score(X_test, y_test):.2f}')

# Print number of mislabeled points
print(f'Number of mislabeled points: {(y_test != logre.predict(X_test)).sum()}')

# Visualize the model with a scatter plot
plt.figure(figsize=(8, 6))

# Scatter plot for actual test data (blue points)
plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual Labels')

# Scatter plot for predicted labels (red stars)
plt.scatter(X_test[:, 0], logre.predict(X_test), color='red', marker='*', label='Predicted Labels')

# Scatter plot for predicted probabilities (green dots)
plt.scatter(X_test[:, 0], logre.predict_proba(X_test)[:, 1], color='green', marker='.', label='Predicted Probabilities')

# Add labels and legend
plt.xlabel('Feature 1')  # Adjust the label based on the actual feature name
plt.ylabel('Label (0: Not Risky, 1: Risky)')
plt.legend()

# Save the plot
plt.savefig('Risky_Loan_Scatter_Plot.png')

# Show the plot
plt.show()


# In[ ]:




