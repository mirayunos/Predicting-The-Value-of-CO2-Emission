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

## Data Understanding

* Getting Data
Using Wget to download the data from IBM Object Storage.

'''
!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv
'''

*Wget is a computer program that retrieves content from web servers.*
