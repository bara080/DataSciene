# Importing necessary libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Loading the Iris dataset
iris = sns.load_dataset("iris")

# Extracting specific columns for petal length and width
X = iris[["petal_length"]]  # Reshape to make X 2D
y = iris["petal_width"]

# Displaying features (X) and target variable (y)
print("Features (X):\n", X)
print("\nTarget variable (y):\n", y)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

# Reshaping the data
X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)

# Creating a Linear Regression model
linearR = LinearRegression()

# Fitting the model to the training data
linearR.fit(X_train, y_train)

# Getting the intercept and coefficient of the model
intercept = linearR.intercept_
coefficient = linearR.coef_

print("Intercept:", intercept)
print("Coefficient:", coefficient)

# Predicting the target variable for training and test data
Y_pred_train = linearR.predict(X_train)
Y_pred_test = linearR.predict(X_test)

# Plotting the training data
plt.scatter(X_train, y_train, color="red", label="Training Data")
plt.plot(X_train, Y_pred_train, color="green", label="Regression Line (Training)")

# Adding labels and title
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Linear Regression on Iris Dataset (Training)")

# Adding legend
plt.legend()

# Display the plot
plt.show()

# Plotting the test data
plt.scatter(X_test, y_test, color="blue", label="Test Data")
plt.plot(X_test, Y_pred_test, color="orange", label="Regression Line (Test)")

# Adding labels and title
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Linear Regression on Iris Dataset (Test)")

# Adding legend
plt.legend()

# Display the plot
plt.show()
