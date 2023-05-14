.. Jaccard Software Manual documentation master file, created by
   sphinx-quickstart on Sat May 13 13:20:30 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Jaccard Software Manual's documentation!
===================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Introduction
------------

This manual explains how to use the code for predicting the stock market based on news headlines.

SetUp PySpark, Read the Data
----------------------------

The first step is to set up PySpark and read the data. This is done with the following code:

.. code-block:: python

   #SetUp PySpark, Read the Data

   import os
   import pyspark
   conf = pyspark.SparkConf()
   sc = pyspark.SparkContext(conf=conf)
   spark = pyspark.sql.SparkSession(sc)
   spark.conf.set("spark.sql.shuffle.partitions", "5")
   spark





For the data file, I am using the Combined_News_DJIA.csv file which is included in the repository
-------------------------------------------------------------





Preprocessing, Encoding Texts into Binary Vector Representation
-------------------------------------------------------------

The next step is to preprocess the data and encode the texts into binary vector representation. This is done with the following code:

.. code-block:: python

   #Preprocessing, Encoding Texts into Binary Vector Representation

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

   # Give the index to each word in our data.
   train_df = train_df.join(count_df, ['word'], 'left').na.drop()
   # Aggregate the indices for each day. "collect_set" function can do this.
   train_df = train_df.groupby('Date').agg(collect_set('index').alias('train'))
   train_df = train_df.withColumnRenamed('Date', 'trainDate')
   # Do the same thing for test set.
   test_df = test_df.join(count_df, ['word'], 'left').na.drop()
   test_df = test_df.groupby('Date').agg(collect_set('index').alias('test'))
   test_df = test_df.withColumnRenamed('Date', 'testDate')data_df = data_df.withColumn('News', col('Top1'))
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

   # Give the index to each word in our data.
   train_df = train_df.join(count_df, ['word'], 'left').na.drop()
   # Aggregate the indices for each day. "collect_set" function can do this.
   train_df = train_df.groupby('Date').agg(collect_set('index').alias('train'))
   train_df = train_df.withColumnRenamed('Date', 'trainDate')
   # Do the same thing for test set.
   test_df = test_df.join(count_df, ['word'], 'left').na.drop()
   test_df = test_df.groupby('Date').agg(collect_set('index').alias('test'))
   test_df = test_df.withColumnRenamed('Date', 'testDate')

Prediction
----------

The final step is to predict the stock market based on the Jaccard similarity between the train and test sets. This is done with the following code:

.. code-block:: python

   #Prediction

   # Since we want to compute the Jaccard Similarity for all train-test pairs, we use "crossJoin".
   merge = train_df.crossJoin(test_df)
   # "array_intersect" and "array_union" are quite similar to set operations in Python.
   # The number of items in intersection divided by the number of items in union is exactly the Jaccard Similarity.
   merge = merge.withColumn('Jaccard', size(array_intersect(col('train'), col('test'))) / size(array_union(col('train'), col('test'))))

   from pyspark.sql.window import Window

   # Use the Window function and pick the one with highest Jaccard Similarity.
   windowSpec = Window.partitionBy('testDate').orderBy(desc('Jaccard'))
   merge = merge.withColumn('rank', rank().over(windowSpec)).where(col('rank')==1)
   # Concatenate the predicted label and ground truth.
   pred = merge.select(col('trainDate'), col('testDate')).join(label_df, merge['trainDate']==label_df['Date'], 'left')
   pred = pred.withColumnRenamed('Label', 'pred').drop('Date')
   pred = pred.join(label_df, merge['testDate']==label_df['Date'], 'left')
   pred = pred.withColumnRenamed('Label', 'true')

   # Finally, check the accuracy.
   acc = pred.where(col('pred')==col('true')).count() / pred.count()
   print(f"Test Accuracy: {acc}")

Conclusion
----------

In this manual, we have shown how to use the code for predicting the stock market based on news headlines. By following the steps outlined in this manual, you can apply this code to your own data and start making predictions.
