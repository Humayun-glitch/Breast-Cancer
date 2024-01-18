# README.md

## Logistic Regression with Polynomial Features

This repository contains a Python script for implementing logistic regression with polynomial features on a breast cancer dataset. The main goals of the script are to demonstrate the process of data preprocessing, feature engineering, model training, and evaluation. The script also includes a visualization of the learning curve for both training and test data using different learning rates.

### Dataset
The dataset used in this project is "cancer_data.csv," which contains information about breast cancer samples. The script reads the CSV file using the Pandas library, drops unnecessary columns ('id' and 'Unnamed: 32'), and converts the 'diagnosis' column to binary values (1 for Malignant, 0 for Benign).

### Polynomial Features and Normalization
The script adds polynomial features to the dataset and performs min-max normalization on the features. Polynomial features help capture non-linear relationships in the data, and normalization ensures that all features are on a similar scale.

### Logistic Regression
The logistic regression model is implemented with a sigmoid activation function. The loss function and gradient descent algorithm are defined to optimize the model parameters. The script visualizes the learning curve for the training data and evaluates the model's performance using a confusion matrix, precision, recall, accuracy, and F1 score.

### Hyperparameter Tuning
The script explores different learning rates to find the optimal value for training the logistic regression model. It iteratively trains the model with various learning rates and plots the F1 score against the learning rates for better visualization.

### How to Use
1. Ensure you have the required libraries installed: NumPy, Pandas, and Matplotlib.
2. Download the breast cancer dataset ("cancer_data.csv") and place it in the same directory as the script.
3. Run the script using a Python environment.

### Results
The script provides insights into the model's performance, showcasing the learning curve for training data and evaluating the F1 score at different learning rates for test data.

### Author
Humayun-Glitch

### Acknowledgments
- The breast cancer dataset used in this project is sourced from cancer_data.csv.
- This script is for educational purposes and can be used as a starting point for understanding logistic regression and hyperparameter tuning.

Feel free to explore, modify, and extend this script for your projects or learning purposes!

###Here are some Images of the running project

![image](https://github.com/Humayun-glitch/Breat-Cancer/assets/57752996/c6ed281d-ea06-468f-89fd-8293029e4335)
![image](https://github.com/Humayun-glitch/Breat-Cancer/assets/57752996/8f3ff29b-8682-46ad-8ae2-8fc473251aa0)
![image](https://github.com/Humayun-glitch/Breat-Cancer/assets/57752996/ed9c6547-133a-4b90-a206-323cd77c55e8)
![image](https://github.com/Humayun-glitch/Breat-Cancer/assets/57752996/aa477d6c-472b-4708-af31-8d57f50bb7d1)

