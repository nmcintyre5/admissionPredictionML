from pyspark.sql import SparkSession

# Create a SparkSession with garbage collection metrics configured
spark = SparkSession.builder \
    .appName("spark") \
    .getOrCreate()

# Read the CSV file into a DataFrame
df = spark.read.csv('Admission_Predict_Ver1.1.csv', header=True, inferSchema=True)

# Drop Serial No Column
df = df.drop('Serial No')

# Show the DataFrame
print("Data Preview: ")
df.show()

# Get the number of rows & columns
print("Dimensions of the data: ",(df.count(),len(df.columns)))

# Print the schema
# print("The columns are: ")
# df.printSchema()

# Get statistics
print("\nDescriptive statistics for the dataset: ")
df.describe().show()

# Create an empty dictionary to store correlations
correlation_dict = {}

# Correlation Analysis
print("Correlation to Chance of Admit column: \n")
for col in df.columns:
    correlation = df.stat.corr('Chance of Admit', col)
    print('{} = {} '.format(col, correlation))
    correlation_dict[col] = correlation

# Sort the dictionary by values (correlation) in descending order
sorted_correlation = dict(sorted(correlation_dict.items(), key=lambda item: item[1], reverse=True))

# Select the top three correlations, skipping the Chance of Admit Column
top_three_correlations = dict(list(sorted_correlation.items())[1:4])

# Print the top three correlations
print("\nTop three correlations (excluding Chance of Admit):\n")
for col, correlation in top_three_correlations.items():
    print('{}: {}'.format(col, correlation))

print("\nThese three have been selected as our features.")

# Feature Selection
# Vector Assembler merges multiple columns into a vector column
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['GRE Score','TOEFL Score','CGPA'],outputCol='features')

# Display dataframe
output_data = assembler.transform(df)
# output_data.show()

# Import Linear Regression and create final data
from pyspark.ml.regression import LinearRegression
final_data = output_data.select('features','Chance of Admit')

#Print schema of final data
#final_data.printSchema()

# Split the dataset into 70% training and 30% testing
train, test = final_data.randomSplit([0.7,0.3])

# Build & train the model
print("\nModel Building & Training:")
models = LinearRegression(featuresCol='features',labelCol='Chance of Admit')
model = models.fit(train)

print("\nThe linear regression model has been trained using 70% of the data.")

# Get coefficients & intercept
print("\nModel coefficients: ",model.coefficients)
print("*Note: Coefficients represent the weights assigned to each feature (GRE Score, TOEFL Score, CGPA). \nIn more simple terms, coefficients tell us how much each factor influences the chance of admission.")

print('\nModel intercept: ',model.intercept)
print("*Note: The intercept represents the expected 'Chance of Admit' when all features are zero.")

# Get summary of the model
summary = model.summary

# Model Performance Summary
print("\nModel Performance Summary:")

# Print the rmse & r2 score
print('\nRoot Mean Squared Error (RMSE):', summary.rootMeanSquaredError)
print("*Note: RMSE measures the average deviation of the predicted values from the actual values. \nA lower RMSE indicates better model performance.")

print('\nR-squared (r2) score:', summary.r2)
print("*Note: R-squared (r2) score indicates the proportion of the variance in the target variable that is predictable from the features. \nA higher R-squared value suggests better model fit and predictive power.\n")

# Transform on the test data
predictions = model.transform(test)

# Display the top 20 rows of predictions
predictions.show(20)

print("\nModel's Performance Evaluation:\n")

# Evaluate the model
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(predictionCol='prediction',labelCol='Chance of Admit',metricName='r2')
print('R2 of the test data:', evaluator.evaluate(predictions))
print("*Note: R-squared, often denoted as r2, is a statistical measure that represents the proportion of the variance in the dependent variable that is explained by the features in a regression model. \nIn simpler terms, it quantifies the goodness of fit of the model to the data.")

'''
# Save the model
model.save("/Users/altuser/Documents/Python Projects/Graduate_Admission_Prediction/model")

# Load the model
from pyspark.ml.regression import LinearRegressionModel
model = LinearRegressionModel.load('model')
'''

# Stop the SparkSession when done
spark.stop()