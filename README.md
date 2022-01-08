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

