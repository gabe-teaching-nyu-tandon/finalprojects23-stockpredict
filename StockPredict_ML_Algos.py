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


stages = []

regexTokenizer = RegexTokenizer(inputCol="comment", outputCol="tokens", pattern="\\W+")
stages += [regexTokenizer]

swr = StopWordsRemover(inputCol="tokens", outputCol="Comments")
stages += [swr]

cv = CountVectorizer(inputCol="Comments", outputCol="token_features", minDF=2.0)#, vocabSize=3, minDF=2.0
stages += [cv]


vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")
stages += [vecAssembler]

[print('\n', stage) for stage in stages]

pipeline = Pipeline(stages=stages)
data = pipeline.fit(df2).transform(df2)

train, test = data.randomSplit([0.7, 0.3])

nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
model = nb.fit(train)

predictions = model.transform(test)
# Select results to view
predictions.limit(20).select("label", "prediction", "probability").show(truncate=False)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
nbaccuracy = evaluator.evaluate(predictions)
print ("Test Area Under ROC: ", nbaccuracy)



# Create ParamGrid and Evaluator for Cross Validation
paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]).build()
cvEvaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
# Run Cross-validation
cv = CrossValidator(estimator=nb, estimatorParamMaps=paramGrid, evaluator=cvEvaluator)
cvModel = cv.fit(train)
# Make predictions on testData. cvModel uses the bestModel.
cvPredictions = cvModel.transform(test)
# Evaluate bestModel found from Cross Validation
evaluator.evaluate(cvPredictions)



# Make predictions on testData. cvModel uses the bestModel.
cvPredictions = cvModel.transform(test)
# Evaluate bestModel found from Cross Validation
print ("Test Area Under ROC: ", evaluator.evaluate(cvPredictions))

log_reg = LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
model2 = log_reg.fit(train)
predictions = model2.transform(test)

evaluator = BinaryClassificationEvaluator().setLabelCol('label').setRawPredictionCol('prediction').setMetricName('areaUnderROC')
lgaccuracy = evaluator.evaluate(predictions)
print(lgaccuracy)

# Create ParamGrid and Evaluator for Cross Validation
paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]).build()
cvEvaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
# Run Cross-validation
cv = CrossValidator(estimator=log_reg, estimatorParamMaps=paramGrid, evaluator=cvEvaluator)
cvModel = cv.fit(train)
# Make predictions on testData. cvModel uses the bestModel.
cvPredictions = cvModel.transform(test)
# Evaluate bestModel found from Cross Validation
evaluator.evaluate(cvPredictions)

