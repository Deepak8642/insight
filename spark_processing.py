from pyspark.sql import SparkSession

# Function to initialize Spark session
def initialize_spark():
    spark = SparkSession.builder.appName('BigDataProcessing').getOrCreate()
    return spark

# Function to load data into Spark DataFrame
def load_data_spark(spark, file_path):
    spark_df = spark.read.csv(file_path, header=True, inferSchema=True)
    return spark_df

# Function to perform data operations on Spark DataFrame
def process_data_spark(spark_df):
    # Drop rows with missing values
    spark_df = spark_df.dropna()

    # Filter data based on a condition
    spark_df = spark_df.filter(spark_df['column_name'] > 0)

    return spark_df

# Function to display Spark DataFrame
def display_spark_df(spark_df):
    spark_df.show()

def main():
    # Initialize Spark session
    spark = initialize_spark()

    # Load data into Spark DataFrame
    file_path = 'preprocessed.csv'
    spark_df = load_data_spark(spark, file_path)

    # Perform data operations
    processed_df = process_data_spark(spark_df)

    # Display Spark DataFrame
    display_spark_df(processed_df)

if __name__ == "__main__":
    main()
