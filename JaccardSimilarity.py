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

