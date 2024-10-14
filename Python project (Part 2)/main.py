#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

import warnings


# In[2]:


warnings.filterwarnings("ignore")


# # Data Analysis

# Dataset source: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

# ## Data Description

# ### Reading the data file

# In[3]:


# reading csv files
data = pd.read_csv('Dry_Bean_Dataset.csv', header=0, sep=",", low_memory=False)
data.head()


# ### Statistics

# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe()


# ### Histograms

# In[7]:


# Plot histograms for numerical features
data.hist(figsize=(10, 8), bins=20)
# Add more vertical space between plots
plt.subplots_adjust(hspace=0.5)

plt.suptitle('Histograms of Numerical Features')
plt.show()


# ### Correlation

# In[8]:


plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")

# Add title and labels
plt.title('Correlation Heatmap')
plt.xlabel('Features')
plt.ylabel('Features')

# Show plot
plt.show()


# In[9]:


g = sns.pairplot(data) #plotting each pair
plt.show()


# ## Showing imbalance

# In[10]:


data['Class'].unique()


# In[11]:


class_counts = data['Class'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(6,5))
plt.bar(class_counts.index, class_counts.values)

plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Distribution of Classes')

plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent clipping of labels

plt.show()


# ## Finding outliers

# In[12]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=data)
plt.title('Boxplot of Features')
plt.xticks(rotation=45)
plt.show()


# ## Data Variance

# In[13]:


data.iloc[:,:-1].var()


# Features with variance values close to zero, such as ShapeFactor1 and ShapeFactor2, can be considered noisy as they exhibit very little variability.

# # Modeling with SVM

# ## Data Preparation

# ### Removing Duplicates

# In[14]:


# Your code here to remove duplicates
data.drop_duplicates()


# ### Converting categorical variables to numerical format

# In[15]:


# Creating a instance of label Encoder.
le = LabelEncoder()
 
# Using .fit_transform function to fit label
# encode and return encoded label
label = le.fit_transform(data['Class'])

print(np.unique(label))


# ### Normalizing numerical features

# In[16]:


# Apply Min-Max to ensure feature uniformity
scaler =  StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(data.iloc[:,:-1]))
scaled_df.columns = data.iloc[:,:-1].columns
scaled_df.index = data.iloc[:,:-1].index
scaled_df.head()


# In[17]:


labels = pd.DataFrame(label,columns=['Class'])
# Concatenate scaled_df and labels along columns
scaled_df = pd.concat([scaled_df, labels], axis=1)
scaled_df.head()


# ## Building the SVM classifier

# In[18]:


# Make a SVM classifier for your chosen dataset, using sklearn as seen during Part 1.
# Default parameters:    C: default=1.0     gamma: default=’scale’     kernel: default=’rbf’

# Split the data into X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(scaled_df.iloc[:,:-1], scaled_df.iloc[:,-1], test_size=0.2, random_state=42)

# Define and Train the SVM Model
classifier = svm.SVC()
classifier.fit(X_train, y_train)

# Evaluate the Model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# ## Hyperparameter tuning

# In[19]:


# Perform hyperparameter tuning to optimize the SVM’s performance.
# Experiment with different parameter combinations and document your process.

# Tune the hyperparameters : C, kernel, gamma
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

# Create an SVM classifier
svm_clf = svm.SVC()

# Perform the grid search on the training data
grid_search = GridSearchCV(estimator=svm_clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Extract results from grid search
results = pd.DataFrame(grid_search.cv_results_)

# Use pivot_table with multi-level index to include 'kernel' in the heatmap
heatmap_data = pd.pivot_table(results, values='mean_test_score', index=['param_C', 'param_gamma'], columns='param_kernel', aggfunc=np.mean)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='flare', fmt='.3f', cbar_kws={'label': 'Accuracy'})

plt.title('Grid Search Results : Accuracy vs Hyperparameter Combinations')
plt.xlabel('Kernel')
plt.ylabel('C and Gamma')
plt.show()

# Get best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Evaluate performance on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", accuracy)


# # Model evaluation
# 
# Accuracy alone can be misleading in imbalanced datasets because it may be high even if the model performs poorly on the minority class.
# 
# In the context of imbalanced data, precision is valuable because it tells us the proportion of correctly predicted positive cases out of all cases predicted as positive. A high precision indicates that when the model predicts a positive case, it is likely to be correct.
# 
# Recall measures the ability of the model to correctly identify all positive instances in the dataset. It is the ratio of true positive predictions to the total number of actual positive instances in the data. In imbalanced datasets, recall is crucial because it highlights the model's ability to capture all instances of the minority class, thus minimizing false negatives.
# 
# The F1-score is the harmonic mean of precision and recall. It provides a balanced measure of the model's performance by considering both false positives and false negatives. F1-score is especially useful in imbalanced datasets because it gives equal weight to precision and recall, making it suitable for evaluating models across different class distributions.

# In[20]:


# Evaluate the SVM model’s performance using standard metrics (e.g., accuracy, precision, recall, F1-score) and justify your choice.

#Calculate performance metrics
# Use weighted average for multiclass classification
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the performance metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# As we can see by the results, tuning the parameters did not have much impact on the performance, this might be due to the imbalancy in the dataset classes, the existence of outliers, or noise in the dataset. 
# 
# To check how these constraints are affecting the performance of the model, we can follow these steps for each constraint:
# 
# 1. Train the model on the original dataset.
# 2. Evaluate the model's performance metrics.
# 3. Remove the constraint from the dataset.
# 4. Retrain the model on the constraint-removed dataset.
# 5. Evaluate the model's performance metrics on the constraint-removed dataset.
# 6. Compare the performance metrics before and after removing the constraint to assess its impact on the model performance.

# ## Influence of outliers

# In[21]:


# Discuss how outliers impact these metrics.
df_X_train = pd.DataFrame(X_train, index=y_train.index, columns = X_train.columns)
df_y_train = pd.DataFrame(y_train)

Q1 = df_X_train.quantile(0.25)
Q3 = df_X_train.quantile(0.75)

# Calculate the IQR for each column
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers detection
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Mark outliers using the lower and upper bounds
outliers_mask = ((df_X_train < lower_bound) | (df_X_train > upper_bound)).any(axis=1)

# Remove outliers from the DataFrame
df_X_train_no_outliers = df_X_train[~outliers_mask]
df_y_train_no_outliers = df_y_train[~outliers_mask]

X_train_no_outliers = df_X_train_no_outliers.iloc[:,:].values
y_train_no_outliers = df_y_train_no_outliers.iloc[:,:].values

# Retrain the model on the outlier-removed dataset
best_model.fit(X_train_no_outliers, y_train_no_outliers.ravel())

y_pred_no_outliers = best_model.predict(X_test)
accuracy_no_outliers = accuracy_score(y_test, y_pred_no_outliers)
precision_no_outliers = precision_score(y_test, y_pred_no_outliers, average='weighted')
recall_no_outliers = recall_score(y_test, y_pred_no_outliers, average='weighted')
f1_no_outliers = f1_score(y_test, y_pred_no_outliers, average='weighted')

# Compare performance metrics before and after removing outliers
print("Performance metrics on original dataset:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("\nPerformance metrics on dataset with outliers removed:")
print("Accuracy:", accuracy_no_outliers)
print("Precision:", precision_no_outliers)
print("Recall:", recall_no_outliers)
print("F1-score:", f1_no_outliers)


# Accuracy: The accuracy on the original dataset (93.46%) is higher than the accuracy on the dataset with outliers removed (87.77%). This indicates that the presence of outliers may have a positive impact on overall accuracy, possibly due to the inclusion of important information from those outliers.
# 
# Precision: Precision measures the proportion of true positive predictions among all positive predictions. In this case, precision is higher on the original dataset (93.51%) compared to the dataset with outliers removed (84.84%). This suggests that the model on the original dataset is better at correctly identifying positive instances, with fewer false positives.
# 
# Recall: Recall measures the proportion of true positive predictions among all actual positive instances. In this case, recall is higher on the original dataset (93.46%) compared to the dataset with outliers removed (87.77%). This suggests that the model on the original dataset is better at capture actual positive instances.
# 
# F1-score: The F1-score is the harmonic mean of precision and recall, providing a balanced measure of a model's performance. The F1-score is higher on the original dataset (93.48%) compared to the dataset with outliers removed (86.06%). This further confirms that the model performs better overall on the original dataset.

# ## Influence of Imbalancy

# ### Using Class Weights
# 
# Assigning different weights to classes can help alleviate the impact of class imbalance.

# In[22]:


# Make the classifier
# Using 'balanced' mode, which automatically adjusts class weights inversely proportional to class frequencies
weighted_classifier = svm.SVC(class_weight='balanced', C=100, gamma='scale', kernel='poly')

# Train them
weighted_classifier.fit(X_train, y_train)

# Get the accuracy on the test set
y_pred_weighted = weighted_classifier.predict(X_test)
accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
precision_weighted = precision_score(y_test, y_pred_weighted, average='weighted')
recall_weighted = recall_score(y_test, y_pred_weighted, average='weighted')
f1_weighted = f1_score(y_test, y_pred_weighted, average='weighted')


# Compare performance metrics before and after giving weight to classes
print("Performance metrics of the original model:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("\nPerformance metrics of the model with class weights:")
print("Accuracy:", accuracy_weighted)
print("Precision:", precision_weighted)
print("Recall:", recall_weighted)
print("F1-score:", f1_weighted)


# ### Using SMOTE

# In[23]:


# Initialize SMOTE object
smote = SMOTE()

# Resample the dataset using SMOTE
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# In[24]:


class_counts = y_train_resampled.value_counts().sort_values(ascending=False)

plt.figure(figsize=(6,5))
plt.bar(class_counts.index, class_counts.values)

plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Distribution of Classes')

plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent clipping of labels

plt.show()


# In[25]:


# Retrain the model on the balanced dataset (using SMOTE)
best_model.fit(X_train_resampled, y_train_resampled)

y_pred_balanced = best_model.predict(X_test)
accuracy_balanced = accuracy_score(y_test, y_pred_balanced)
precision_balanced = precision_score(y_test, y_pred_balanced, average='weighted')
recall_balanced = recall_score(y_test, y_pred_balanced, average='weighted')
f1_balanced = f1_score(y_test, y_pred_balanced, average='weighted')

# Compare performance metrics before and after removing imbalancy
print("Performance metrics on original dataset:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("\nPerformance metrics on the balanced dataset:")
print("Accuracy:", accuracy_balanced)
print("Precision:", precision_balanced)
print("Recall:", recall_balanced)
print("F1-score:", f1_balanced)


# We tried two methods of removing imbalancy, yet none of them improved the results compared to the perforomance of the tuned model on the original dataset.

# ## Influence of Noise

# ### PCA
# 
# We try to improve the results by implementing PCA on the dataset. 
# 
# Principal Component Analysis (PCA) is used to: 
# 
# a) denoise 
# 
# b) reduce dimensionality 
# 
# It does not eliminate noise, but can reduce it.

# In[26]:


# Tune the hyperparameters : n_components
n = np.arange(1,17)
param_grid = {
    'n_components': n,
}

# Create a PCA object
pca = PCA()

# Perform the grid search on the training data
grid_search = GridSearchCV(pca, param_grid)
grid_search.fit(X_train, y_train)

# Display the best parameters
print("Best Parameters:", grid_search.best_params_)


# As the best number of components that we get is very close to the number of features we already have, there will not be any significant changes in our results.

# In[29]:


# Create a PCA object with 2 components
pca = PCA(n_components=15)
# Fit the PCA model to our dataset and Transform our dataset into the new lower-dimensional space
X_reduced = pca.fit_transform(X_train)

# Train the best classifier with the reduced data
best_model.fit(X_reduced, y_train)

# Transform the test dataset using the same PCA transformation
X_test_reduced = pca.transform(X_test)
# Get the accuracy on the test set
y_pred_pca = best_model.predict(X_test_reduced)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
precision_pca = precision_score(y_test, y_pred_pca, average='weighted')
recall_pca = recall_score(y_test, y_pred_pca, average='weighted')
f1_pca = f1_score(y_test, y_pred_pca, average='weighted')

# Compare performance metrics before and after applying PCA
print("Performance metrics on original dataset:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("\nPerformance metrics on the dataset after applying PCA:")
print("Accuracy:", accuracy_pca)
print("Precision:", precision_pca)
print("Recall:", recall_pca)
print("F1-score:", f1_pca)


# ### Feature Engineering
# 
# By generating polynomial features from existing features we try to capture nonlinear relationships in the data. Feature engineering can help improve the discriminatory power of the model and make it more robust to noise.

# In[32]:


#degree-2 polynomial features
poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(X_train)

# Train the best classifier with the reduced data
best_model.fit(X_poly, y_train)

# Get the accuracy on the test set
# Transform the test dataset using the same PCA transformation
X_test_poly =  poly.transform(X_test)
y_pred_poly = best_model.predict(X_test_poly)
accuracy_poly = accuracy_score(y_test, y_pred_poly)
precision_poly = precision_score(y_test, y_pred_poly, average='weighted')
recall_poly = recall_score(y_test, y_pred_poly, average='weighted')
f1_poly = f1_score(y_test, y_pred_poly, average='weighted')

# Compare performance metrics before and after generating polynomial features
print("Performance metrics on original dataset:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("\nPerformance metrics on the dataset after generating polynomial features:")
print("Accuracy:", accuracy_poly)
print("Precision:", precision_poly)
print("Recall:", recall_poly)
print("F1-score:", f1_poly)


# None of the methods we used were able to make any improvements on the results. 
