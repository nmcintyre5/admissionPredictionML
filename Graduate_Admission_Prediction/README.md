# Graduate Admission Predictor to Unicorn University
[Overview](#overview) | [Key Features](#key-features) | [How to Install](#how-to-install) | [Credits](#credits)

## Overview

This script will walk you through a build of a linear regression model using PySpark ML to predict student admissions at Unicorn University. It uses the graduate admission dataset obtained from [GitHub User Education454](https://github.com/education454/admission_dataset/blob/master/Admission_Predict_Ver1.1.csv). It utilizes a Simple Linear Regression machine learning algorithm from the PySpark Machine learning library. Please note that the dataset and the model in this project are for learning purposes only and cannot be used in real-life scenarios.

## Key Features

- **Model Building:** The first Python script (`modelBuild.py`) will guide the user through the model building process using PySpark and Linear Regression. It will print descriptive statistics and analyze model performance, specifically RMSE and r2.
- **User Prediction:** The second Python script (`admissionPrediction.py`) allows users to input their GRE score, TOEFL score, and CGPA to predict their likelihood of admission.

## How to Install

1. **Clone the Repository:** Clone this repository to your local machine.
2. **Install Dependencies:** Make sure you have Python 3.x and PySpark installed on your system.
3. **Run the Scripts:** Execute the Python scripts to perform model building and user prediction. No need to download the dataset separately as it's included in the repository.

## Usage

### Script 1: Model Building Walkthrough

```bash
python modelBuild.py
```
This script walks the user through the model build process using PySpark and Linear Regression. It prints the descriptive statistics of the data, performs a correlation analysis, selects the top three features, and builds and trains a linear regression model. Then, it prints the model's performance summary and analyzes how well the model performs, utilizing Root Mean Squared Error (RMSE) and R-squared (r2).

### Script 2: User Prediction
```bash
python admissionPrediction.py
```
This script allows users to input their GRE score, TOEFL score, and CGPA to predict their likelihood of admission to Unicorn University.

## Credits
This script is based on the project "Graduate Admission Prediction with PySpark ML" from the Coursera Project Network.

