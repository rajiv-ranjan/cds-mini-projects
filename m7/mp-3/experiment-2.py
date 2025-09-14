# Import necessary libraries
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import col, when
from pyspark.mllib.stat import Statistics
from operator import add
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# 1. Create Spark Session and load the data
# Start spark session
spark = SparkSession.builder.appName("ComplexAnalytics").getOrCreate()

# Access sparkContext from sparkSession instance.
sc = spark.sparkContext
sqlContext = SQLContext(sc)

# Load the dataset and show the top 10 records
filePath = "kddcup.data_10_percent.gz"
raw_data = sc.textFile(filePath)

# Display top 10 records
print("--- Top 10 Raw Records ---")
for line in raw_data.take(10):
    print(line)
print("-" * 25)

# 2. RDD Basic Operations
# Convert the data to CSV format (list of elements).
csv_data = raw_data.map(lambda line: line.split(","))

# Count how many interactions are normal and attacked in the dataset.
normal_data = csv_data.filter(lambda x: x[41] == "normal.")
attack_data = csv_data.filter(lambda x: x[41] != "normal.")

normal_count = normal_data.count()
attack_count = attack_data.count()

print(f"Number of normal interactions: {normal_count}")
print(f"Number of attack interactions: {attack_count}")
print("-" * 25)

# Protocol and Service combinations using Cartesian product
protocols = csv_data.map(lambda x: x[1]).distinct()
services = csv_data.map(lambda x: x[2]).distinct()
product = protocols.cartesian(services)
print(f"All Protocol-Service combinations: {product.count()}")
print("--- Top 5 combinations ---")
print(product.take(5))
print("-" * 25)


# Inspecting interaction duration
# Normal interactions duration stats
normal_duration_data = normal_data.map(lambda x: int(x[0]))
total_normal_duration = normal_duration_data.reduce(add)
mean_normal_duration = total_normal_duration / normal_count
print(f"Total duration for normal interactions: {total_normal_duration}")
print(f"Mean duration for normal interactions: {mean_normal_duration:.2f}")

# Attack interactions duration stats
attack_duration_data = attack_data.map(lambda x: int(x[0]))
total_attack_duration = attack_duration_data.reduce(add)
mean_attack_duration = total_attack_duration / attack_count
print(f"Total duration for attack interactions: {total_attack_duration}")
print(f"Mean duration for attack interactions: {mean_attack_duration:.2f}")
print("-" * 25)


# Data aggregation with key/value pair RDDs
key_value_duration = csv_data.map(lambda x: (x[41], int(x[0])))
duration_by_key = key_value_duration.reduceByKey(add)
print("--- Duration by Intrusion Type ---")
for key, val in duration_by_key.collect():
    print(f"{key}: {val}")
print("-" * 25)


# 3. Create a DataFrame with the header as features
# Read the features (kddcup.names) and preprocess.
with open('kddcup.names', 'r') as f:
    # Skip the first line which contains attack types
    f.readline()
    # Read feature names and types
    feature_lines = f.readlines()

# Extract feature names and clean them
col_names = [line.split(':')[0] for line in feature_lines]
# Add the target column name
col_names.append('intrusion_type')


# Create a dataframe with the data and headers
kdd_df = spark.createDataFrame(csv_data, col_names)
print("--- DataFrame Schema ---")
kdd_df.printSchema()
print("--- Top 5 DataFrame Records ---")
kdd_df.show(5)
print("-" * 25)

# What is the count of each protocol type?
protocol_counts = kdd_df.groupBy('protocol_type').count()
print("--- Protocol Type Counts ---")
protocol_counts.show()
print("-" * 25)

# 4. Register the DataFrame as a temporary table and extract data using queries
kdd_df.createOrReplaceTempView("connections")

# Query to extract the label and their frequencies
label_counts = sqlContext.sql("SELECT intrusion_type, COUNT(*) as frequency FROM connections GROUP BY intrusion_type ORDER BY frequency DESC")
print("--- Intrusion Type Frequencies ---")
label_counts.show()
print("-" * 25)

# Select the distinct protocol types with their count of transactions which are not normal
attack_by_protocol = sqlContext.sql("""
    SELECT protocol_type, COUNT(*) as attack_count
    FROM connections
    WHERE intrusion_type != 'normal.'
    GROUP BY protocol_type
    ORDER BY attack_count DESC
""")
print("--- Attacks by Protocol Type ---")
attack_by_protocol.show()
print("-" * 25)


# Select count of transactions in each protocol type that lasts more than 1 second (duration > 1000), with no data transfer from destination (dst_bytes == 0)
long_duration_no_transfer = sqlContext.sql("""
    SELECT protocol_type, COUNT(*) as transaction_count
    FROM connections
    WHERE duration > 1000 AND dst_bytes == 0
    GROUP BY protocol_type
    ORDER BY transaction_count DESC
""")
print("--- Long Duration & No Dest. Transfer by Protocol ---")
long_duration_no_transfer.show()
print("-" * 25)


# 5. Find the highly correlated columns
# Identify columns which are not integer type and remove those columns
numeric_cols = [c for c, t in kdd_df.dtypes if t not in ['string']]
numeric_rdd = kdd_df.select(numeric_cols).rdd.map(lambda row: [int(c) for c in row])

# Apply correlation function on the data
correlation_matrix = Statistics.corr(numeric_rdd, method="pearson")

# Create a dataframe with correlation matrix
corr_df = pd.DataFrame(correlation_matrix, columns=numeric_cols, index=numeric_cols)
corr_df.fillna(0, inplace=True)

# Get the highly correlated features
threshold = 0.8
upper_tri = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
highly_correlated = [column for column in upper_tri.columns if any(upper_tri[column].abs() > threshold)]
correlated_pairs = upper_tri[highly_correlated].stack().reset_index()
correlated_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
correlated_pairs_filtered = correlated_pairs[correlated_pairs['Correlation'].abs() > threshold]

print(f"--- Feature pairs with correlation > {threshold} ---")
print(correlated_pairs_filtered)
print("-" * 25)

# 6. Analysis Report
# Find the ratio of attacked transactions vs normal transactions
ratio = attack_count / normal_count
print(f"Ratio of attacked to normal transactions: {ratio:.2f}")
print("-" * 25)

# Describe the statistics of attacked and normal transactions
print("--- Statistics for Normal Transactions ---")
kdd_df.filter(col('intrusion_type') == 'normal.').describe().show()
print("--- Statistics for Attacked Transactions ---")
kdd_df.filter(col('intrusion_type') != 'normal.').describe().show()

# Select any two features that influence the intrusion_type and visualize the scatter plot
# Let's pick 'src_bytes' and 'dst_bytes' and sample the data for plotting
df_sample = kdd_df.select(
    col("src_bytes").cast("float"),
    col("dst_bytes").cast("float"),
    (when(col("intrusion_type") == "normal.", 1).otherwise(0)).alias("is_normal")
).sample(withReplacement=False, fraction=0.01, seed=42)

pandas_df = df_sample.toPandas()

print("--- Visualizing Feature Separation ---")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='src_bytes',
    y='dst_bytes',
    hue='is_normal',
    data=pandas_df,
    palette=['red', 'green'],
    alpha=0.5
)
plt.title('Scatter Plot of Source Bytes vs. Destination Bytes')
plt.xlabel('Source Bytes')
plt.ylabel('Destination Bytes')
plt.xscale('log')
plt.yscale('log')
plt.legend(title='Transaction Type', labels=['Attack', 'Normal'])
plt.grid(True)
plt.show()

# Stop the Spark session
spark.stop()
