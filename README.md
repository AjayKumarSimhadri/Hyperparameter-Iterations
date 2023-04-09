# Hyperparameter-Iterations
Using ML Flow Performing a Model and Load it into ML API during Multiple Hyperparameter Iteration.

The code reads the 'train.csv' file from the Titanic dataset on Kaggle using Pandas library and drops the columns 'PassengerId', 'Name', 'Cabin', and 'Ticket' using the drop() method. The 'Age' column is filled with the mean age value of the dataset, while the 'Embarked' column is filled with the mode value of the 'Embarked' column. The 'Sex' column is mapped to numerical values, with 'male' mapped to 0 and 'female' mapped to 1. Finally, the 'Embarked' column is converted to dummy variables using the get_dummies() method.

The X and y variables are then defined for use in training and testing the logistic regression model. X contains all the columns except the 'Survived' column, while y contains only the 'Survived' column.

The train_test_split() function from scikit-learn is used to split the data into training and validation sets, with 80% of the data used for training and 20% used for validation.

A parameter grid is defined for the logistic regression model, with the 'C' parameter taking values of 0.01, 0.1, 1, and 10, and the 'penalty' parameter taking values of 'l1' and 'l2'. A logistic regression model is initialized with a random state of 42.

GridSearchCV is used to find the best combination of hyperparameters for the logistic regression model. The grid search is performed with 5-fold cross-validation. The best estimator is then used to predict the labels of the validation set, and the accuracy of the model is calculated using the accuracy_score() function.

Finally, the best hyperparameters and accuracy metric are logged using the MLflow library, and the best model is logged using the mlflow.sklearn.log_model() function.
