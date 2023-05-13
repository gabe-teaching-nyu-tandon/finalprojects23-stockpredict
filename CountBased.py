#!/usr/bin/env python
# coding: utf-8

# # CSGY-6513 Big Data Final Project
# In this notebook, I implement the Count-Based model on Kaggle Dataset.

# ## 1. Setting Up PySpark, Read the Data.

# In[1]:



def count_based_algo(news_data):
    import os
    import pyspark

    conf = pyspark.SparkConf()
    #conf.set('spark.ui.proxyBase', '/user/' + os.environ['JUPYTERHUB_USER'] + '/proxy/4041')
    conf.set('spark.sql.repl.eagerEval.enabled', True)
    conf.set('spark.driver.memory', '4g')
    sc = pyspark.SparkContext(conf=conf)
    spark = pyspark.SQLContext.getOrCreate(sc)


    # In[ ]:

    from pyspark.sql.functions import col, lower, regexp_replace, to_date, concat_ws, split, trim, explode, udf, lit, log, sum, min

    data_df = spark.read.format('csv').option('inferSchema','true').option('header','true').load(news_data)
    data_df = data_df.withColumn('Date', to_date('Date'))


    # In[ ]:


    data_df.show(3)


    # For each day, we have label and 25 news headlines. The label is 1 if the DJIA (Dow Jones Industrial Average) daily return is plus, and 0 if minus.

    # ## 2. Preprocessing, Computing the Score for Each Frequent Word.

    # In[ ]:

    
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


    # In[ ]:


    # Perform the word counting on training data.
    count_df = train_df.select(col('word')).where(col('Label')==1).groupBy('word').count()
    count_df = count_df.withColumnRenamed('count', 'pos')
    neg_df = train_df.select(col('word')).where(col('Label')==0).groupBy('word').count()
    neg_df = neg_df.withColumnRenamed('count', 'neg')
    count_df = count_df.join(neg_df, ['word'], 'outer')
    count_df = count_df.na.fill(value=0)
    count_df = count_df.withColumn('sum', col('pos')+col('neg'))

    # We only focus on frequent words, which is top 900 words in this case.
    count_df = count_df.where(col('sum')>100)
    print(f"Number of Words: {count_df.count()}")


    # In[ ]:


    # Since the number of positives (the DJIA daily return is plus) and negatives are not the same, we normalize it.
    # Multiplying by 100 doesn't have a specific meaning. Just scaling so that the values don't get too small.
    pos_sum = count_df.agg({'pos': 'sum'}).collect()[0][0]
    neg_sum = count_df.agg({'neg': 'sum'}).collect()[0][0]
    count_df = count_df.withColumn('pos', col('pos') * 100 / pos_sum)
    count_df = count_df.withColumn('neg', col('neg') * 100 / neg_sum)
    count_df.sort(col('sum').desc()).show(10)


    # These are the top 10 frequent words. However, since pos and neg values are nearly the same, the final score will be close to 0, meaning these words have limited influence on our prediction.

    # In[ ]:


    from pyspark.sql.types import DoubleType

    def clip(val, floor, ceiling):
        if val < floor:
            val = floor
        elif val > ceiling:
            val = ceiling
        return val
    clip_udf = udf(clip, DoubleType())

    # We lower bound with small value to avoid having log(0). Upper bound doesn't have any meaning but we need to feed some value to the function.
    lower, upper = 1e-7, 100
    count_df = count_df.withColumn('pos', clip_udf(col('pos'), lit(lower), lit(upper)))
    count_df = count_df.withColumn('neg', clip_udf(col('neg'), lit(lower), lit(upper)))

    # Computing the score. Since less frequent values tend to get high score, we bound it.
    count_df = count_df.withColumn('score', log(col('pos')) - log(col('neg')))
    lower, upper = -0.5, 0.5
    count_df = count_df.withColumn('score', clip_udf(col('score'), lit(lower), lit(upper)))


    # Below is the words with top 10 highest scores, which means they show up more in the positive contexts (Days when DJIA rose).

    # In[ ]:


    count_df.sort(col('score').desc()).show(10)


    # Similarly, below is the words with top 10 lowest scores.

    # In[ ]:


    count_df.sort(col('score')).show(10)


    # ## 3. Prediction

    # In[ ]:


    # For each word in test data, we assign the computed score by left join.
    test_df = test_df.join(count_df.select(col('word'), col('score')), ['word'], 'left')
    test_df = test_df.na.fill(value=0, subset=['score'])


    # In[ ]:


    # Use groupBy to aggregate the word scores.
    # min('Label') is just keeping one label (ground truth) for each day. Taking minimum is not necessary since the labels are the same for a certain day.
    test_df = test_df.select(col('Date'), col('Label'), col('score')).groupBy('Date').agg(min('Label'),sum('score'))
    test_df = test_df.withColumnRenamed('min(Label)', 'Label').withColumnRenamed('sum(score)', 'sum')


    # In[ ]:


    # Finally, compute the accuracy.
    acc = test_df.where((col('Label')==1)==(col('sum')>0)).count() / test_df.count()
    print(f"Test Accuracy: {acc}")
    return acc


# In[ ]:

count_based_algo('Combined_News_DJIA.csv')



# In[ ]:




