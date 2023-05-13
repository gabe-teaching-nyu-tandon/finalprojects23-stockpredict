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