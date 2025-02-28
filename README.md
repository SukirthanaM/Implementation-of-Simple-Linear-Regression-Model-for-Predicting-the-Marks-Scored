# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SUKIRTHANA.M
RegisterNumber: 212224220112

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datfile
df.head()
df.tail()
#Segregating data to variables
x = df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,y_pred,color="green")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
*/
```

## Output:
#To read first 5 elements
![image](https://github.com/user-attachments/assets/c0fc7ff5-14b7-4a87-ae74-669e6a26a0bd)
#To read last 5 elements
![image](https://github.com/user-attachments/assets/b20ca50a-c1d7-472e-be9a-0540774682ed)
#The code selects all rows and all columns except the last one from the DataFrame df, then converts the result into a NumPy array and stores it in x.
![image](https://github.com/user-attachments/assets/2dcbdce6-2d71-4477-912c-55952ddf9ddf)
#The code selects all rows from the second column of the DataFrame df, converts it into a NumPy array, and stores it in the variable y
![image](https://github.com/user-attachments/assets/c0944415-3827-4b9a-a0b3-b755a1d1b2a5)
#The code splits the feature set x and target variable y into training and testing sets, where 1/3 of the data is allocated to testing, and 2/3 to training.The resulting subsets are stored in x_train, x_test, y_train, and y_test.
![image](https://github.com/user-attachments/assets/79edf0be-791c-47cd-a1bc-27f5421b7d87)
#The code initializes a linear regression model, trains it using the training data (x_train, y_train), uses the trained model to predict target values for the test data (x_test), and then returns or prints the predicted values in y_pred.
![image](https://github.com/user-attachments/assets/17b8d03e-10a0-479a-8fac-f630e6434ad9)
#y_test contains the actual target values for the test dataset, and it is used to assess the accuracy of the model's predictions.
![image](https://github.com/user-attachments/assets/80bc0880-a4a4-4963-8119-143c453a7149)
#code computes three common regression evaluation metrics
![image](https://github.com/user-attachments/assets/7cdb7b03-28fd-427c-a485-bd28353ef39f)
#The code generates a plot that shows the relationship between the training data (x_train for "Hours" and y_train for "Scores") as a scatter plot. 
![image](https://github.com/user-attachments/assets/c9d3cd82-2da6-4e1b-bf54-1952423c2341)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
