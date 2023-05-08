# **Predicting the movement of the stock price using Sentiment Analysis** 



## Team Members: 

- Jash Unadkat (jru5640)
- Jiazhao Shi (js12624)
- Naveen Kumar Mallemala (nm3937)
- Yichen Lee (yl9396) 



## Background

The stock market has always been a subject of great interest to investors, financial analysts, and researchers alike. Its inherent volatility and unpredictability make it a challenging area to study, and predicting stock price movements has been an ongoing quest for many. Recent advancements in Big Data and Artificial Intelligence have paved the way for novel approaches to tackle this complex problem, one of which is the use of Sentiment Analysis.

Sentiment Analysis, a subfield of Natural Language Processing, is the process of identifying and extracting opinions, emotions, and attitudes from textual data. In the context of stock market prediction, sentiment analysis has been employed to gauge the market sentiment based on the information available in news articles, social media posts, and other online sources. The underlying assumption is that public opinion, as expressed through these sources, has an impact on stock prices and their movements.

Several research studies have reported promising results by incorporating sentiment analysis into stock market prediction models, thereby substantiating its potential in this domain. However, the vast amount of unstructured data generated on a daily basis calls for the use of Big Data techniques to effectively handle, process, and analyze the data. Moreover, there is an opportunity to explore the performance of various Big Data models to identify the most suitable approach for predicting stock price movements.



## Overview

The stock market is a dynamic and ever-changing environment that requires investors to make informed decisions based on various factors that affect the stock price. One of the key factors that influence stock prices is investor sentiment, which can be measured by analyzing the text data from social media platforms such as Reddit and Twitter. In this project, our primary aim is to forecast stock price trends using Sentiment Analysis.
Data Acquisition: To achieve this, we plan to collect data from Reddit, Twitter, and Yahoo Finance through APIs such as PushshiftAPI, Praw, and Tweepy. The Reddit and Twitter data will contain comments and posts from specific dates, while Yahoo Finance data will represent the S&P 500 stock price changes for particular dates, displayed as positive or negative percentages. Our ultimate goal is to identify a correlation and rationale behind the S&P 500 price increases or decreases using the text data.
Prepossessing: We expect to collect over a million data points, which will require significant preprocessing. Firstly, we will convert the S&P 500 percentage change to a label, with negative percentages assigned label 0 and positive percentages receiving label 1. We will also use Big Data techniques such as Bloom Filter to determine whether a post is related to a stock or not, and PySpark N-grams to convert all the sentences in a post to n-grams and remove unnecessary words before passing the data to the model.
Comparative Analysis: Once the data has been preprocessed, we will use PySpark to implement our models, including a Count-Based approach, Jaccard Similarity Search, and some Machine Learning classifiers. These models will require additional Big Data techniques such as PySpark window, Cartesian Join, MongoDB, and Spark ML libraries.
The Count-Based approach involves counting the number of times a word appears in the text and comparing it with the frequency of the same word in a predefined set of positive and negative sentiment words. This approach is simple and effective but does not 





## Architecture Diagram


[Please Click Here](https://drive.google.com/file/d/1MPIQGLnW9QhFf8dUt1a1oq7XwBErhy2A/view?usp=sharing)



## Organization and Components

In this section, we will describe how data will be fetched and preprocessed before being used to train our models.

### 3.1 Data Collection 

We plan to fetch social media posts from Reddit and Twitter. To collect data from Reddit, we will use the PushshiftAPI and Praw API. For Twitter, we will utilize the Tweepy API. For both Reddit and Twitter data, we will fetch the post ID, post title, comments related to the post, and timestamp of when the post or comment was created.

Additionally, we will fetch stock data from Yahoo Finance using the Yahoo Finance API. We will obtain the stock price for each stock on specific dates and calculate the difference from the previous date to determine the percentage increase or decrease.

Subsequently, we will merge the social media posts and stock data.



### 3.2 Preprocessing

Before feeding the data to the models, it needs to be preprocessed. We plan to use a Bloom Filter to quickly and efficiently filter unrelated posts, and PySpark ML to create n-grams and remove unnecessary words and characters.



#### 3.2.1 Bloom Filter

A bloom filter is a space-efficient, key-value storage system that provides extremely fast filtering to determine if an element is part of a set. It can allow false positives but not false negatives. We will use it in our project to determine if a post is related to a stock or not. We will do so by first going through all post titles and adding the IDs of titles containing certain keywords related to the stock in the bloom filter. Then, we will go through all of the social media data and filter out comments and posts based on the IDs in the bloom filter.



#### 3.2.2 PySpark ML 

Before passing the data to the model, all sentences in a post will need to be converted to n-grams, and all unnecessary words will need to be removed. To accomplish this, we will utilize PySpark ML's NGram and StopWordsRemover modules.



## Methods and Interfaces

### **CountBased** Approach:

The CountBased interface will be responsible for implementing the count-based approach, which is a method to quantify the sentiment of text data. This interface will analyze the frequency of positive and negative words in the collected comments and posts, and calculate a sentiment score for each data point. It will take preprocessed text data as input and return sentiment scores.



### **CreateRedditDataset**:

The CreateRedditDataset interface will manage the collection and initial preprocessing of Reddit data. It will fetch data using the PushshiftAPI and Praw, and convert the obtained data into a structured format, ready for further preprocessing.



### **JaccardSimilarity**:

The JaccardSimilarity interface will be responsible for implementing the Jaccard Similarity Search, which measures the similarity between text data points based on their shared words. This interface will take preprocessed text data as input and return similarity scores between data points.



### **LocalitySensitiveHashing**:

The LocalitySensitiveHashing interface will implement the Locality Sensitive Hashing technique, which is used to efficiently find approximate nearest neighbors in high-dimensional spaces. This interface will be utilized in the Jaccard Similarity Search to quickly find similar data points, reducing the computational complexity of the search.

### **ML-Based Approach**:

Several Machine Learning algorithms will be implemented to load the data, pre-process it, apply various transformations, train/test/validate, evaluate and compare the results (accuracies). 




## TimeLine

### **4/30/2023 - 5/2/2023: Initial Setup and Data Pre-Processing**

- Jash Unadkat (jru5640) & Jiazhao Shi (js12624): Intial setup & Clean and preprocess Reddit and Twitter data, employing Big Data techniques such as Bloom Filter, PySpark N-grams, and Broadcast Variable.
- Naveen Kumar Mallemala (nm3937) & Yichen Lee (yl9396): Intial setup & Convert S&P 500 percentage changes into binary labels and merge with the preprocessed text data.

### **5/3/2023 - 5/6/2023: Model Implementation**

- Jash Unadkat (jru5640): Implement the Count-Based approach using PySpark.
- Jiazhao Shi (js12624): Develop the Jaccard Similarity Search and integrate it with the Locality Sensitive Hashing technique.
- Naveen Kumar Mallemala (nm3937) & Yichen Lee (yl9396): Implement Machine Learning classifiers using Spark ML libraries.


### **5/6/2023 - 5/10/2023: CI/CD Pipeline and Testcases**

- Jash Unadkat (jru5640) & Jiazhao Shi (js12624): Setup and implement CI/CD pipeline to automate the work.
- Naveen Kumar Mallemala (nm3937) & Yichen Lee (yl9396): Write testcases for the implemented approches and do the testing. 

### **5/10/2023 - 5/11/2023: Model Evaluation and Optimization**

- All team members: Evaluate the performance of each model and fine-tune the parameters to achieve optimal accuracy and optimize the code wherever possible. Compare the results of the different models to provide insights for future research.

### **5/12/2023 - 5/14/2023: Documentation and Report Preparation**

- All team members: Document the implemented models, techniques, and findings. Prepare a comprehensive project report detailing the methodology, results, and conclusions.

### **5/15/2023: Final Review and Submission**

- All team members: Present the project to stakeholders and submit the final report and code.