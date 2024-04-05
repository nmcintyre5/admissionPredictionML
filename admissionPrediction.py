from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Read HTML content from file
# with open("admissionPredictor.html", "r") as file:
#    html_content = file.read()

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Admission Predictor") \
    .getOrCreate()

# Load the data
data = spark.read.csv('Admission_Predict_Ver1.1.csv', header=True, inferSchema=True)

# Prepare the data for modeling
assembler = VectorAssembler(inputCols=["GRE Score", "TOEFL Score", "CGPA"], outputCol="features")
data = assembler.transform(data).select("features", "Chance of Admit")

# Train the model
lr = LinearRegression(labelCol="Chance of Admit", featuresCol="features")
model = lr.fit(data)

# Function to predict admission chance
def predict_admission_chance(gre_score, toefl_score, cgpa):
    # Create a DataFrame with user input
    user_data = spark.createDataFrame([(gre_score, toefl_score, cgpa)], ["GRE Score", "TOEFL Score", "CGPA"])
    # Transform user data to match model input format
    user_data = assembler.transform(user_data).select("features")
    # Make prediction
    prediction = model.transform(user_data).select(col("prediction").alias("Chance of Admit")).collect()[0]["Chance of Admit"]
    return prediction

# Main program
if __name__ == "__main__":
    # Get user input
    gre_score = float(input("Enter your GRE score (260-340): "))
    toefl_score = float(input("Enter your TOEFL score (0-120): "))
    cgpa = float(input("Enter your CGPA (0-4): "))

    # Normalize CGPA to a scale out of 10
    cgpa = cgpa / 4 * 10

    # Predict admission chance
    admission_chance = predict_admission_chance(gre_score, toefl_score, cgpa)

    # Convert predicted probability to percentage
    admission_chance_percentage = round(admission_chance * 100, 2)

    # Display prediction
    print("Your predicted chance of admission is:", admission_chance_percentage, "%")

    # Stop Spark session