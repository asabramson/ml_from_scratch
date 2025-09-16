# Machine & Deep Learning From Scratch

This repository contains various machine and deep learning architectures built from scratch. Helper libraries such as `NumPy` were used, however, libraries built specifically for machine learning were not utilized, as the purpose of this project is to act as an alternative to such libraries. The project is broken up into 3 parts, with the first being the implementation of 9 machine learning model architectures, the second being the implementation of neural networks and various layer types, and the third being a frontend web application which users can interact with the models using their own data to train and test.

## PART 1 - Machine Learning Models

Below is a list of the featured architectures, which includes the name of the architecture, the type of learning (supervised vs unsupervised, regression vs classification vs clustering), and a brief description of what the architecture does. 

| Name | Type | Brief Description |
| ----------- | ----------- | ----------- |
| Linear Regression | Supervised, Regression | Linear Regression models a linear relationship between a dependent variable (also known as target/output) and one or more independent variables (also known as features or inputs). The model finds a best-fit line through the data points which can be used to predict additional outputs. |
| Logistic Regression | Supervised, binary classification or regression* | A classification algorithm that predicts probabilities using the sigmoid function, then classifies inputs into two categories based on a threshold. |
| K Nearest Neighbors (KNN) | Supervised, classification or regression | Classifies data points by looking at the majority class among their k nearest neighbors using distance metrics like Euclidean distance. |
| K Means Clustering | Unsupervised, clustering | Groups data into k clusters by iteratively assigning points to the nearest cluster center and recalculating the centers until convergence. |
| Principal Component Analysis (PCA) | Unsupervised, feature/dimension reduction | A dimensionality reduction technique that finds new axes principal components to maximize variance, simplifying high-dimensional data while retaining most of its information. |
| Naive Bayes | Supervised, classification | A probabilistic classifier that applies Bayes’ theorem with the assumption that features are independent, making it simple yet effective for many text and categorical datasets. |
| Support Vector Machine (SVM) | Supervised, classification | Finds the optimal hyperplane that separates data into classes by maximizing the margin between the closest points of different classes. |
| Decision Tree | Supervised, classification or regression* | Splits the data into branches based on feature thresholds, creating a tree structure that makes decisions by following paths from the root to a leaf node. |
| Random Forest | Supervised, classification or regression* | An ensemble of many decision trees trained on random subsets of data and features, combining their predictions through majority voting for improved accuracy and stability. |

\*NOTE: Although Logistic Regression, Decision Trees, and Random Forests can be used for regression, I have been built for classification tasks specifically here, and would require modifications to be able to do both.


## PART 2 - Deep Learning/Neural Networks
### WORK IN PROGRESS

## PART 3 - Frontend Web Application
### WORK IN PROGRESS