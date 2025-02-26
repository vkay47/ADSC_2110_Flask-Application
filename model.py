# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Reading the dataset
dataset = pd.read_csv("final_housing.csv")

# Extracting the features and the target variable
x = dataset.iloc[:,1:6]
y = dataset.iloc[:,1]

# Creating the regression object
regressor = LinearRegression()

# Training the regression object with the data
regressor.fit(x, y)

# Saving the trained model to disk
with open("model.pkl", "wb") as file:
    pickle.dump(regressor, file)

# Making predictions using the trained model
y_pred = regressor.predict(x)

# Plotting the regression line

plt.title("Price vs Area ")
plt.xlabel("Area")
plt.ylabel("Price")
sns.regplot(x=x.iloc[:,1],y=y,
               scatter_kws={"color": "black"},
               line_kws={"color": "red"}, ci=None)
plt.savefig("static/regression_line.png")
# plt.show()
