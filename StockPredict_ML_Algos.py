from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, udf, concat_ws, concat, to_date, collect_list, translate, regexp_replace, when
from pyspark.sql.types import BooleanType, StringType
from bloom_filter2 import BloomFilter
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes, LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib.pyplot as plt
import numpy as np

spark = SparkSession.builder.appName('SentimentAnalyzer').getOrCreate()

# Load data and rename column
df = spark.read.format("csv")\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .option("multiLine", "true")\
    .option("delimiter", "¥")\
    .load("data/reddit-data.csv")\
    .coalesce(5)


#Preprocessing

df = df.withColumn('comment', lower(col('comment')))

# filter to see if title column contains any keyword from keywords
keywords = ["SP500" , "S&P500"]
def my_filter(col):
    for keyword in keywords:
        if keyword.lower() in col.lower():
            return True
    return False

filterUDF = udf(my_filter, BooleanType())
ids = df.filter(col("title").isNotNull()).filter(filterUDF('title')).select("ID")

# create and populate bloom filter
bloomFilterIDS = BloomFilter(ids.count(), 0.000000001)
collected_ids = ids.collect()
for row in collected_ids:
    bloomFilterIDS.add(row["ID"])


broadcastFilterIds = spark.sparkContext.broadcast(bloomFilterIDS)

def my_filter_by_ids(col):
    return col in broadcastFilterIds.value
        
filterIdUDF = udf(my_filter_by_ids, BooleanType())
bloomedFilteredData = df.filter(col("SP500").isNotNull()).filter(filterIdUDF('ID'))
bloomedFilteredData = bloomedFilteredData.withColumn("date_stock",to_date("timestamp"))
bloomedFilteredData = bloomedFilteredData.na.drop(subset=["comment"])
bloomedFilteredData= bloomedFilteredData.drop("_c0","id","title", "timestamp", "time_key", "TESLA")
df1 = bloomedFilteredData.groupby('date_stock', 'SP500').agg(collect_list('comment').alias("comment"))
df2 = df1.withColumn("comment",
   concat_ws(",",col("comment")))

df2 = df2.withColumn('comment', translate('comment', '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~', '" '))
df2 = df2.withColumn('comment', regexp_replace('comment', '"', ' '))
df2 = df2.withColumn('comment', regexp_replace('comment', "'", ' '))
df2.filter(df2.date_stock == "2022-05-04") \
    .show(truncate=False)
df2= df2.withColumn("SP500", when(df2["SP500"]>0,1).otherwise(0))
df2= df2.withColumnRenamed("SP500","label")