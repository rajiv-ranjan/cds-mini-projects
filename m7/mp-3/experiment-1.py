# @title Install packages and Download Dataset
# !pip -qq install pyspark
# Download the data
# !wget -qq https://cdn.iisc.talentsprint.com/CDS/Datasets/kddcup.data_10_percent.gz
# Download feature names
# !wget -qq https://cdn.iisc.talentsprint.com/CDS/Datasets/kddcup.names
# print("Successfully Installed packages and downloaded datasets!")

from utility import download_and_unzip

download_and_unzip(
    filename="kddcup.data_10_percent.gz",
    url="https://cdn.iisc.talentsprint.com/CDS/Datasets/kddcup.data_10_percent.gz",
)

download_and_unzip(
    filename="kddcup.names",
    url="https://cdn.iisc.talentsprint.com/CDS/Datasets/kddcup.names",
)

print("Data downloaded successfully")

### Create Spark Session and load the data (1 point)

#### Import required packages
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.mllib.stat import Statistics
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from operator import add

#### Create a Spark session
# Start spark session
spark = SparkSession.builder \
    .appName("KDDCup99 Analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

#### Creating an RDD from a File
sc = spark.sparkContext

# Load the dataset and show the top 10 records
filePath = "/content/kddcup.data_10_percent.gz"
raw_data = sc.textFile(filePath)
print("Top 10 records:")
print(raw_data.take(10))

### RDD Basic Operations (4 points)

#### Convert the data to CSV format (list of elements).
csv_data = raw_data.map(lambda line: line.split(","))
print("First 5 CSV records:")
print(csv_data.take(5))

# Count how many interactions are normal and attacked in the dataset.
normal_count = csv_data.filter(lambda x: x[-1] == "normal.").count()
attacked_count = csv_data.filter(lambda x: x[-1] != "normal.").count()

print(f"Normal interactions: {normal_count}")
print(f"Attacked interactions: {attacked_count}")
print(f"Total interactions: {csv_data.count()}")

#### Protocol and Service combinations using Cartesian product
protocols = csv_data.map(lambda x: x[1]).distinct()
services = csv_data.map(lambda x: x[2]).distinct()

print("Protocol types:", protocols.collect())
print("Service types:", services.collect())

# Now let's do the Cartesian product
protocol_service_pairs = protocols.cartesian(services)
print("Protocol-Service pairs (first 10):")
print(protocol_service_pairs.take(10))

#### Inspecting interaction duration
# For normal interactions
normal_durations = csv_data.filter(lambda x: x[-1] == "normal.").map(lambda x: int(x[0]))
normal_duration_sum = normal_durations.reduce(add)
normal_duration_mean = normal_duration_sum / normal_count

# For attacked interactions
attacked_durations = csv_data.filter(lambda x: x[-1] != "normal.").map(lambda x: int(x[0]))
attacked_duration_sum = attacked_durations.reduce(add)
attacked_duration_mean = attacked_duration_sum / attacked_count

print(f"Normal duration - Sum: {normal_duration_sum}, Mean: {normal_duration_mean:.2f}")
print(f"Attacked duration - Sum: {attacked_duration_sum}, Mean: {attacked_duration_mean:.2f}")

#### Data aggregation with key/value pair RDDs
# Create key/value pairs of intrusion type and duration
intrusion_duration_pairs = csv_data.map(lambda x: (x[-1], int(x[0])))

# Calculate total duration for each intrusion type
total_duration_by_intrusion = intrusion_duration_pairs.reduceByKey(add)
print("Total duration by intrusion type:")
print(total_duration_by_intrusion.collect())

### Create a DataFrame with the header as features (2 points)

# Read the features and preprocess
with open("/content/kddcup.names", "r") as f:
    features_content = f.readlines()

# Process feature names
features = []
for line in features_content[1:]:  # Skip the first line (intrusion types)
    if ":" in line:
        feature_name = line.split(":")[0].strip()
        features.append(feature_name)

# Add the intrusion_type column at the end
features.append("intrusion_type")
print("Features:", features)

# Create a dataframe with the data and headers
from pyspark.sql.types import *

# Define schema
schema_fields = []
for i, feature in enumerate(features):
    if i in [0, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]:
        schema_fields.append(StructField(feature, IntegerType(), True))
    else:
        schema_fields.append(StructField(feature, StringType(), True))

schema = StructType(schema_fields)

# Create DataFrame
df = spark.createDataFrame(csv_data, schema)
print("DataFrame schema:")
df.printSchema()
print("Total records:", df.count())

# What is the count of each protocol type?
protocol_counts = df.groupBy("protocol_type").count()
protocol_counts.show()

#### Register the DataFrame as a temporary table and extract the data using queries
df.createOrReplaceTempView("network_intrusion")

# Query to extract the label and their frequencies
label_freq = spark.sql("SELECT intrusion_type, COUNT(*) as frequency FROM network_intrusion GROUP BY intrusion_type ORDER BY frequency DESC")
label_freq.show()

# Select distinct protocol types with their count of transactions which are not normal
non_normal_protocols = spark.sql("""
    SELECT protocol_type, COUNT(*) as count 
    FROM network_intrusion 
    WHERE intrusion_type != 'normal.' 
    GROUP BY protocol_type 
    ORDER BY count DESC
""")
non_normal_protocols.show()

# Select count of transactions in each protocol type that lasts more than 1 second with no data transfer from destination
long_duration_no_transfer = spark.sql("""
    SELECT protocol_type, COUNT(*) as count 
    FROM network_intrusion 
    WHERE duration > 1000 AND dst_bytes = 0 
    GROUP BY protocol_type 
    ORDER BY count DESC
""")
long_duration_no_transfer.show()

### Find the highly correlated columns (2 points)

# Identify numeric columns
numeric_columns = [f.name for f in df.schema.fields if isinstance(f.dataType, IntegerType)]
print("Numeric columns:", numeric_columns)

# Convert to RDD of vectors for correlation calculation
numeric_rdd = df.select(numeric_columns).rdd.map(lambda row: [float(x) for x in row])

# Calculate correlation matrix
corr_matrix = Statistics.corr(numeric_rdd, method="pearson")

# Create DataFrame with correlation matrix
corr_df = pd.DataFrame(corr_matrix, index=numeric_columns, columns=numeric_columns)

# Fill null values with 0
corr_df = corr_df.fillna(0)

# Find highly correlated features (correlation > 0.8)
high_corr_pairs = []
for i in range(len(corr_df.columns)):
    for j in range(i+1, len(corr_df.columns)):
        if abs(corr_df.iloc[i, j]) > 0.8:
            high_corr_pairs.append((corr_df.columns[i], corr_df.columns[j], corr_df.iloc[i, j]))

print("Highly correlated feature pairs (correlation > 0.8):")
for pair in high_corr_pairs:
    print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")

### Analysis report (1 points)

# Find the ratio of attacked transactions vs normal transactions
total_count = df.count()
attack_ratio = attacked_count / total_count
normal_ratio = normal_count / total_count

print(f"Attack ratio: {attack_ratio:.3f}")
print(f"Normal ratio: {normal_ratio:.3f}")
print(f"Attack to Normal ratio: {attack_ratio/normal_ratio:.3f}")

# Describe statistics of attacked and normal transactions
normal_stats = df.filter(df.intrusion_type == "normal.").describe(["duration", "src_bytes", "dst_bytes"])
attacked_stats = df.filter(df.intrusion_type != "normal.").describe(["duration", "src_bytes", "dst_bytes"])

print("Normal transaction statistics:")
normal_stats.show()

print("Attacked transaction statistics:")
attacked_stats.show()

# Select two features and visualize scatter plot
# Convert to Pandas for visualization
sample_df = df.sample(False, 0.1).toPandas()  # Sample 10% for visualization

# Create binary label (1 for normal, 0 for attacked)
sample_df['label'] = sample_df['intrusion_type'].apply(lambda x: 1 if x == 'normal.' else 0)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(sample_df['src_bytes'], sample_df['dst_bytes'], 
                     c=sample_df['label'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Label (1=Normal, 0=Attacked)')
plt.xlabel('Source Bytes')
plt.ylabel('Destination Bytes')
plt.title('Source vs Destination Bytes (Normal vs Attacked)')
plt.xscale('log')
plt.yscale('log')
plt.show()

# Another visualization with different features
plt.figure(figsize=(10, 6))
scatter = plt.scatter(sample_df['duration'], sample_df['count'], 
                     c=sample_df['label'], cmap='coolwarm', alpha=0.6)
plt.colorbar(scatter, label='Label (1=Normal, 0=Attacked)')
plt.xlabel('Duration')
plt.ylabel('Count')
plt.title('Duration vs Count (Normal vs Attacked)')
plt.show()

# Stop Spark session
spark.stop()