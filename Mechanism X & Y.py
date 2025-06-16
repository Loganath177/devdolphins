# Databricks notebook source
# MAGIC %md
# MAGIC **Mechanism X**

# COMMAND ----------

# MAGIC %md
# MAGIC Create a mechanism X to invoke every second and create a chunk of next 10,000 transaction entries from GDrive and put them into a S3 folder.

# COMMAND ----------

#Set AWS Credentials
spark.conf.set("fs.s3a.access.key", "************************")
spark.conf.set("fs.s3a.secret.key", "**************************")
spark.conf.set("fs.s3a.endpoint", "s3.amazonaws.com")

# COMMAND ----------

#File path
#dbfs:/FileStore/CustomerImportance.csv
#/dbfs/FileStore/CustomerImportance.csv
#dbfs:/FileStore/transactions.csv
#/dbfs/FileStore/transactions.csv

# COMMAND ----------

#install gdown for private files
%pip install gdown

# COMMAND ----------

import gdown

file_id = "************************"
url = f"https://drive.google.com/uc?id={file_id}"

# Download the file to a temporary location
gdown.download(url, "/tmp/transactions.csv", quiet=False)

# COMMAND ----------

#Download customer file from gd
import gdown

file_id = "***************************"
url = f"https://drive.google.com/uc?id={file_id}"

# Download the file to a temporary location
gdown.download(url, "/tmp/CustomerImportance.csv", quiet=False)

# COMMAND ----------

import pandas as pd
import boto3
import time
from datetime import datetime

bucket_name = "devdolphin-logu"
s3 = boto3.client('s3',
    aws_access_key_id="**********************",
    aws_secret_access_key="*************************"
)

df = pd.read_csv("/tmp/transactions.csv")
chunk_size = 10000

for i in range(0, len(df), chunk_size):
    chunk = df[i:i+chunk_size]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"transactions_chunk_{ts}_{i//chunk_size}.csv"
    local_path = f"/tmp/{file_name}"
    s3_key = f"input_trans/{file_name}"

    chunk.to_csv(local_path, index=False)
    s3.upload_file(local_path, bucket_name, s3_key)

    print(f"Uploaded {file_name} to S3.")
    time.sleep(1)


# COMMAND ----------

# MAGIC %md
# MAGIC **Mechanism Y**

# COMMAND ----------

# MAGIC %md
# MAGIC Create a mechanism Y that starts at the same time as X and ingests the above S3 stream as soon as transaction chunk files become available, detects the below patterns asap and puts these detections to S3 , 50 at a time to a unique file. Each detection consists of YStartTime(IST), detectionTime(IST),patternId, ActionType, customerName, MerchantId. Whichever fields for a given detection aren't applicable, leave them as “” empty string.

# COMMAND ----------

import boto3

# Replace these with your actual credentials and bucket details
aws_access_key_id = "*************************"
aws_secret_access_key = "*******************************"
bucket_name = "devdolphin-logu"
s3_key = "customer/customerimportance.csv"  # key = path inside the bucket

# Set up S3 client
s3 = boto3.client('s3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Upload file to S3
local_path = "/tmp/CustomerImportance.csv"
s3.upload_file(local_path, bucket_name, s3_key)

print(f"File uploaded to s3://{bucket_name}/{s3_key}")


# COMMAND ----------

# MAGIC %md
# MAGIC PatId1 - A customer in the top 1 percentile for a given merchant for the total number of transactions with the bottom 1% percentile weight, merchant wants to UPGRADE(actionType) them. Upgradation only begins once total transactions for the merchant exceed 50K.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC PatId2 - A customer whose average transaction value for a given merchant < Rs 23 and made at least 80 transactions with that merchant, merchant wants to mark them as CHILD(actionType) asap.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC PatId3 - Merchants where number of Female customers < number of Male customers overall and number of female customers > 100, are marked DEI-NEEDED(actionType) 
# MAGIC

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from datetime import datetime
import pytz

spark = SparkSession.builder.getOrCreate()

# 1. Define schemas
transaction_schema = StructType([
    StructField("step", IntegerType()),
    StructField("customer", StringType()),
    StructField("age", IntegerType()),
    StructField("gender", StringType()),
    StructField("zipcode", StringType()),
    StructField("merchant", StringType()),
    StructField("zipMerchant", StringType()),
    StructField("category", StringType()),
    StructField("amount", DoubleType()),
    StructField("fraud", IntegerType())
])

cust_schema = StructType([
    StructField("Source", StringType()),
    StructField("Target", StringType()),
    StructField("Weight", DoubleType()),
    StructField("typeTrans", StringType()),
    StructField("fraud", StringType())
])

# 2. Read CustomerImportance statically
cust_imp = spark.read \
    .option("header", True) \
    .schema(cust_schema) \
    .csv("s3://devdolphin-logu/customer/customerimportance.csv")

# 3. Define batch logic
def detect_patterns(batch_df, batch_id):
    if batch_df.isEmpty():
        return

    ist_time = datetime.now(pytz.timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S')

    # Cache batch
    batch_df.cache()

    # ── Pattern 1: UPGRADE ────────────────────────────────
    merchant_txn_count = batch_df.groupBy("merchant").agg(count("*").alias("merchant_txn_count"))

    cust_merchant_freq = batch_df.groupBy("customer", "merchant")\
        .agg(count("*").alias("cust_txn_count"))

    filtered_txn = cust_merchant_freq.join(merchant_txn_count, "merchant")\
        .filter("merchant_txn_count >= 50000")

    w1 = Window.partitionBy("merchant").orderBy(col("cust_txn_count").desc())
    top_1 = filtered_txn.withColumn("txn_percentile", percent_rank().over(w1))\
        .filter("txn_percentile <= 0.01")

    w2 = Window.partitionBy("typeTrans").orderBy("Weight")
    bottom_1 = cust_imp.withColumn("weight_percentile", percent_rank().over(w2))\
        .filter("weight_percentile <= 0.01")

    upgrade = top_1.alias("t").join(
        bottom_1.alias("i"),
        (col("t.customer") == col("i.Source")) & (col("t.merchant") == col("i.Target")),
        "inner"
    ).selectExpr(
        f"'{ist_time}' as YStartTime",
        f"'{ist_time}' as detectionTime",
        "'PatId1' as patternId",
        "'UPGRADE' as ActionType",
        "t.customer as customerName",
        "t.merchant as MerchantId"
    )

    # ── Pattern 2: CHILD ────────────────────────────────────
    child = batch_df.groupBy("customer", "merchant")\
        .agg(
            avg("amount").alias("avg_amount"),
            count("*").alias("txn_count")
        ).filter("txn_count >= 80 AND avg_amount < 23")\
        .selectExpr(
            f"'{ist_time}' as YStartTime",
            f"'{ist_time}' as detectionTime",
            "'PatId2' as patternId",
            "'CHILD' as ActionType",
            "customer as customerName",
            "merchant as MerchantId"
        )

    # ── Pattern 3: DEI-NEEDED ──────────────────────────────
    gender_agg = batch_df.groupBy("merchant", "gender")\
        .agg(countDistinct("customer").alias("cust_count"))

    gender_pivot = gender_agg.groupBy("merchant")\
        .pivot("gender", ["Male", "Female"])\
        .agg(first("cust_count"))\
        .na.fill(0)\
        .withColumnRenamed("Male", "male_count")\
        .withColumnRenamed("Female", "female_count")

    dei = gender_pivot.filter("female_count > 100 AND female_count < male_count")\
        .selectExpr(
            f"'{ist_time}' as YStartTime",
            f"'{ist_time}' as detectionTime",
            "'PatId3' as patternId",
            "'DEI-NEEDED' as ActionType",
            "'' as customerName",
            "merchant as MerchantId"
        )

    final_df = upgrade.unionByName(child, allowMissingColumns=True)\
                      .unionByName(dei, allowMissingColumns=True)

    # ── Write in 50-record chunks to S3 ─────────────────────
    if final_df.count() > 0:
        output_path = f"s3a://devdolphin-logu/pattern_detections/{ist_time.replace(':', '_').replace(' ', '_')}/"
        final_df.repartition(50).write.mode("overwrite").json(output_path)

# 4. Start streaming from S3 input
query = (
    spark.readStream
         .option("header", True)
         .schema(transaction_schema)
         .csv("s3a://devdolphin-logu/input_trans/")
         .writeStream
         .foreachBatch(detect_patterns)
         .outputMode("append")
         .start()
)

query.awaitTermination()


# COMMAND ----------

