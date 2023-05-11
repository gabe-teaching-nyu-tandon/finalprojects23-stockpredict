#SetUp PySpark, Read the Data

import os
import pyspark

#conf = pyspark.SparkConf()
#conf.set('spark.ui.proxyBase', '/user/' + os.environ['JUPYTERHUB_USER'] + '/proxy/4041')
#conf.set('spark.sql.repl.eagerEval.enabled', True)
#conf.set('spark.driver.memory', '4g')
#sc = pyspark.SparkContext(conf=conf)
#spark = pyspark.SQLContext.getOrCreate(sc)

conf = pyspark.SparkConf()

sc = pyspark.SparkContext(conf=conf)
spark = pyspark.sql.SparkSession(sc)
spark.conf.set("spark.sql.shuffle.partitions", "5")
spark

from pyspark.sql.functions import *

news_data = 'Combined_News_DJIA.csv'
data_df = spark.read.format('csv').option('inferSchema','true').option('header','true').load(news_data)
data_df = data_df.withColumn('Date', to_date('Date'))
label_df = data_df.select(col('Date'), col('Label'))

#Preprocessing, Encoding Texts into Binary Vector Representation
# Concatenate all the news headlines into column "News" for each day.
data_df = data_df.withColumn('News', col('Top1'))
for i in range(1, 26):
    data_df = data_df.withColumn('News', concat_ws(' ', 'News', 'Top'+str(i)))

# Lowercase them, remove letters that is not numbers and alphabets, and split them into words.
data_df = data_df.withColumn('News', regexp_replace(lower(col('News')), r"[^0-9a-z]", " "))
data_df = data_df.withColumn('News', split(col('News'), " "))

# Discard unnecessary columns and prepare for the word count using "explode" function.
data_df = data_df.select(col('Date'), col('Label'), col('News'))
data_df = data_df.withColumn('News', explode(col('News'))).withColumnRenamed('News', 'word')
data_df = data_df.withColumn('word', trim(col('word')))
# Remove the row with empty string and "b". All the texts start with a letter "b" which is nothing to do with the news headlines.
data_df = data_df.where((col('word')!='') & (col('word')!='b'))

# Split the data into training and test sets.
train_df = data_df.where(col('Date') < '2015-09-17')
test_df = data_df.where(col('Date') >= '2015-09-17')

# We pick only the frequent words to form binary vectors.
count_df = train_df.select(col('word')).where(col('Label')==1).groupBy('word').count()
count_df = count_df.where(col('count')>=50).select(col('word'))

# Instead of storing the actual binary vectors, we keep the index of entries with 1.
# So, give index for each word.
count_df = count_df.withColumn('index', monotonically_increasing_id()+1)

print(f"Number of Words Selected: {count_df.count()}")