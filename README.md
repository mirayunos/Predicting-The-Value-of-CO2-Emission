# Predicting-The-Value-of-CO2-Emission
This is my first Data Science Project. The purpose of this project is to predict the value of CO2 emission by using Simple Linear Regression algorithm. Special thanks to Dr. Mohammed Al-Obaydee for the guidance.

## Business Understanding

Case Study: To find the value of CO2 emission from cars.

* Goal: To predict the value of CO2 emission based on single independent variable.
* Objective: To build a predictive model using Simple Linear Regression algorithm.

*Single Independent Variable means it stands alone and cannot be changed.*

## Analytic Approach

Simple Linear Regression algorithm is used to predict the value of CO2 emission from cars.

*Simple Linear Regression algorithm is a supervised machine learning algorithm to model linear relationship between to variables.*

## Data Requirements

* CO2 Emission
* Engine Size
* Fuel Consumption

## Data Collection

The dataset is sourced from Canada.ca.
Datasets provide model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada.

## Data Understanding and Preparation

* Getting Data
Using Wget to download the data from IBM Object Storage.

```
!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv
```

*Wget is a computer program that retrieves content from web servers.*



#### `FuelConsumption.csv`:

We have downloaded a fuel consumption dataset, **`FuelConsumption.csv`**, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. [Dataset source](http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64?cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork-20718538&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork-20718538&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork-20718538&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork-20718538&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)

-   **MODELYEAR** e.g. 2014
-   **MAKE** e.g. Acura
-   **MODEL** e.g. ILX
-   **VEHICLE CLASS** e.g. SUV
-   **ENGINE SIZE** e.g. 4.7
-   **CYLINDERS** e.g 6
-   **TRANSMISSION** e.g. A6
-   **FUEL CONSUMPTION in CITY(L/100 km)** e.g. 9.9
-   **FUEL CONSUMPTION in HWY (L/100 km)** e.g. 8.9
-   **FUEL CONSUMPTION COMB (L/100 km)** e.g. 9.2
-   **CO2 EMISSIONS (g/km)** e.g. 182   --> low --> 0

* Importing Packages

```
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline
```

* Reading the Data

```
df = pd.read_csv("FuelConsumption.csv")
df.head()
```

* Descriptive Statistics Exploration 

```
# To summarize the data
df.describe()
```

**Features such as Engine-Size, Cylinders, FuelConsumption_Comb and CO2 Emission are explored further.

```
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)
```

* Exploratory Data Analysis (EDA)

```
#Plotting for visualisation

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()
```
![download](https://user-images.githubusercontent.com/93753467/148644637-99e3c45f-fc51-4e12-9e0b-e10cc839bcdd.png)


Plotting selected features to see linearity.

```
#Cylinders vs CO2 Emission

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("CYLINDERS")
plt.ylabel("Emission")
plt.show()
```
![download (1)](https://user-images.githubusercontent.com/93753467/148644649-3ad90229-986e-4ad1-8aaf-f1ec806d1b13.png)


```
#Fuel Consumption_Combustion vs CO2 Emission

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()
```
![download (2)](https://user-images.githubusercontent.com/93753467/148644668-2485d68c-d34c-4404-9613-2f4210104698.png)


```
#Engine Size vs CO2 Emission

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
```
![download (3)](https://user-images.githubusercontent.com/93753467/148644729-a51588a8-a40b-4eb7-91fa-5948b7e8cca2.png)

Engine size vs CO2 Emission shows stronger positive correlation compared to the other features.



* Train-Test Split Method

Creating an out-of-sample testing by splitting the dataset:
* 80% of the entire data for training
* 20% for testing

*For the main code, 0.75 of dataset is selected instead to show the difference of the outcomes.*

We create a mask to select random rows using np.random.rand() function.

```
msk = np.random.rand(len(df)) < 0.80
#The numpy.random.rand() function creates an array of specified shape and fills it with random values.

train = cdf[msk]
test = cdf[~msk]
```

*The train-test split method is used to estimate the performance of machine learning algorithms when they are used to make predictions on data that are not used to train the model.*

## Modelling

* Simple Regression Model

Linear Regression fits a linear model with coefficients B = (B1, ..., Bn) to minimize the 'residual sum of squares' between the independent x in the dataset, and the dependent y by the linear approximation.

* Training Data Distribution

```
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
```
![download (4)](https://user-images.githubusercontent.com/93753467/148745548-8243f02c-7018-4ce2-aa2d-3c72db2cd5f3.png)

Using scikit-learn package to model data.

```
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
```

Coefficients:  [[39.43589096]]
Intercept:  [124.30848132]

As mentioned before, Coefficient and Intercept in the simple linear regression are the parameters of the fitted line. Given that it is a simple linear regression with only 2 parameters, and knowing that the parameters are the intercept and slope of the line, scikit-learn can estimate them directly from our data. Notice that all of the data must be available to traverse and calculate the parameters.

* Plot Output

Plotting fitted line over data distribution.

```
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r') #-r is for the colour of the line, can be change to -g, you will see green
plt.xlabel("Engine size")
plt.ylabel("Emission")
```
![download (5)](https://user-images.githubusercontent.com/93753467/148746093-6f277d9a-680d-4228-8c2b-2496f27011b9.png)

## Evaluation

Mean Squared Error (MSE) is used to calculate the accuracy of our model based on the test set.

*Mean Squared Error (MSE) is the mean of the squared error. It is more popular than Mean Absolute Error because the focus is geared more towards the large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.

```
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )
```

Result:

Mean absolute error: 24.43
Residual sum of squares (MSE): 981.65
R2-score: 0.77

*R squared is a popular metric to check for accuracy of the model. It represents how close the data are to the fitted regression line. The higher the R squared, the better the model fits the data. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).*
