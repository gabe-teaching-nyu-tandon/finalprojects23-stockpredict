#Import required libraries for the project

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

#Instantiate a Spark Session
spark = SparkSession.builder.appName('SentimentAnalyzer').getOrCreate()

# Load data
df = spark.read.format("csv")\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .option("multiLine", "true")\
    .option("delimiter", "¥")\
    .load("/Users/mallenesha/SESCProject/group_git_sandbox-stockpredict/StockPredict_Data.csv")\
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

#Spark ML pipeline setup

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


#Training and testing models
#Pipeline Fitting:
pipeline = Pipeline(stages=stages)
data = pipeline.fit(df2).transform(df2)

train, test = data.randomSplit([0.7, 0.3])

#Naive Bayes Implementation
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
model = nb.fit(train)

predictions = model.transform(test)
# Select results to view
predictions.limit(20).select("label", "prediction", "probability").show(truncate=False)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
nbaccuracy = evaluator.evaluate(predictions)
print ("Test Area Under ROC: ", nbaccuracy)

#Cross Validation Evaluation for Naive Bayes Model:

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

#Logistic regression Model:
log_reg = LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
model2 = log_reg.fit(train)
predictions = model2.transform(test)

evaluator = BinaryClassificationEvaluator().setLabelCol('label').setRawPredictionCol('prediction').setMetricName('areaUnderROC')
lgaccuracy = evaluator.evaluate(predictions)
print(lgaccuracy)


#Cross Validation Evaluation for logistic Rergression Model
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

#Random Forest Classifier Model:
rf = RandomForestClassifier().setLabelCol('label').setFeaturesCol('features').setNumTrees(10)
model = rf.fit(train)
predictions = model.transform(test)

evaluator = BinaryClassificationEvaluator().setLabelCol('label').setRawPredictionCol('prediction').setMetricName("areaUnderROC")
rfaccuracy = evaluator.evaluate(predictions)
print(rfaccuracy)


#Cross Validation Evaluation for Randon Forest Classifier Model
# Create ParamGrid and Evaluator for Cross Validation
paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]).build()
cvEvaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
# Run Cross-validation
cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=cvEvaluator)
cvModel = cv.fit(train)
# Make predictions on testData. cvModel uses the bestModel.
cvPredictions = cvModel.transform(test)
# Evaluate bestModel found from Cross Validation
evaluator.evaluate(cvPredictions)

#Decision Tree Classifier Model:
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
dtModel = dt.fit(train)
predictions = dtModel.transform(test)

evaluator = BinaryClassificationEvaluator().setRawPredictionCol('prediction')
#evaluator = BinaryClassificationEvaluator(labelCol="label", featuresCol="features", maxDepth=2)
dtAccuracy = evaluator.evaluate(predictions)
print(dtAccuracy) 


#Cross Validation Evaluation for Decision Tree Clasifier
# Create ParamGrid and Evaluator for Cross Validation
paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]).build()
cvEvaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
# Run Cross-validation
cv = CrossValidator(estimator=dt, estimatorParamMaps=paramGrid, evaluator=cvEvaluator)
cvModel = cv.fit(train)
# Make predictions on testData. cvModel uses the bestModel.
cvPredictions = cvModel.transform(test)
# Evaluate bestModel found from Cross Validation
evaluator.evaluate(cvPredictions)

#Support Vector Classifier Model:
# Define your classifier
lsvc = LinearSVC(maxIter=10, regParam=0.1)

# Fit the model
lsvcModel = lsvc.fit(train)

# Compute predictions for test data
predictions = lsvcModel.transform(test)

# Define the evaluator method with the corresponding metric and compute the classification error on test data
evaluator = BinaryClassificationEvaluator().setRawPredictionCol('prediction')
svmaccuracy = evaluator.evaluate(predictions) 

# Show the accuracy
print("Test accuracy = %g" % (svmaccuracy))

#Cross Validation Evaluation of Support Vector Classifier
# Create ParamGrid and Evaluator for Cross Validation
paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]).build()
cvEvaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
# Run Cross-validation
cv = CrossValidator(estimator=lsvc, estimatorParamMaps=paramGrid, evaluator=cvEvaluator)
cvModel = cv.fit(train)
# Make predictions on testData. cvModel uses the bestModel.
cvPredictions = cvModel.transform(test)
# Evaluate bestModel found from Cross Validation
evaluator.evaluate(cvPredictions)
