import unittest
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

class SentimentAnalyzerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Instantiate a Spark Session
        cls.spark = SparkSession.builder.appName('SentimentAnalyzerTest').getOrCreate()
        
        # Load test data
        cls.test_data = cls.spark.createDataFrame([
            ("comment1", 1),
            ("comment2", 0),
            ("comment3", 1),
            ("comment4", 0)
        ], ["comment", "label"])

        # Preprocessing
        
        # Regex Tokenizer
        regex_tokenizer = RegexTokenizer(inputCol="comment", outputCol="tokens", pattern="\\W+")
        
        # Stop Words Remover
        stop_words_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
        
        # Count Vectorizer
        count_vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="features")
        
        # Vector Assembler
        vector_assembler = VectorAssembler(inputCols=["features"], outputCol="assembled_features")
        
        # Naive Bayes Classifier
        naive_bayes = NaiveBayes(smoothing=1.0, modelType="multinomial")
        
        # Create a pipeline
        cls.pipeline = Pipeline(stages=[
            regex_tokenizer,
            stop_words_remover,
            count_vectorizer,
            vector_assembler,
            naive_bayes
        ])

        # Fit the pipeline on the test data
        cls.model = cls.pipeline.fit(cls.test_data)
    

    def test_prediction_accuracy(self):
        # Make predictions on the test data
        predictions = self.model.transform(self.test_data)
        
        # Evaluate the accuracy of the predictions
        evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
        accuracy = evaluator.evaluate(predictions)
        
        # Assert that the accuracy is greater than or equal to 0.5
        self.assertGreaterEqual(accuracy, 0.5)
    
    def test_cross_validation(self):
        # Define the parameter grid for cross-validation
        param_grid = ParamGridBuilder().addGrid(self.model.stages[-1].smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]).build()
        
        # Define the evaluator for cross-validation
        evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
        
        # Create a cross-validator
        cross_validator = CrossValidator(estimator=self.model, estimatorParamMaps=param_grid, evaluator=evaluator)
        
        # Fit the cross-validator on the test data
        cv_model = cross_validator.fit(self.test_data)
        
        # Make predictions on the test data using the best model from cross-validation
        cv_predictions = cv_model.transform(self.test_data)
        
        # Evaluate the accuracy of the predictions
        accuracy = evaluator.evaluate(cv_predictions)
        
        # Assert that the accuracy is greater than or equal to 0.5
        self.assertGreaterEqual(accuracy, 0.5)

    @classmethod
    def setUpClass(cls):
        # Create a SparkSession for testing
        cls.spark = SparkSession.builder \
            .appName('SentimentAnalyzerTest') \
            .master('local[2]') \
            .getOrCreate()


    def test_data_preprocessing(self):
        # Load the test data
        test_data = [
            (1, "Title 1", "2022-05-04", "SP500", "Comment 1"),
            (2, "Title 2", "2022-05-04", "S&P500", "Comment 2"),
            (3, "Title 3", "2022-05-05", "SP500", "Comment 3"),
        ]
        df = self.spark.createDataFrame(test_data, ["ID", "title", "date_stock", "SP500", "comment"])

        # Apply data preprocessing
        df_processed = preprocess_data(df)

        # Assert the processed DataFrame has the expected columns and values
        expected_columns = ["ID", "date_stock", "comment", "label"]
        self.assertListEqual(df_processed.columns, expected_columns)

        expected_values = [
            (1, "2022-05-04", "comment 1", 1),
            (2, "2022-05-04", "comment 2", 1),
            (3, "2022-05-05", "comment 3", 0),
        ]
        self.assertListEqual(df_processed.collect(), expected_values)

    def test_model_training(self):
        # Create a test DataFrame with preprocessed data
        test_data = [
            (1, "2022-05-04", "comment 1", 1),
            (2, "2022-05-04", "comment 2", 1),
            (3, "2022-05-05", "comment 3", 0),
        ]
        df = self.spark.createDataFrame(test_data, ["ID", "date_stock", "comment", "label"])

        # Train the model
        model = train_model(df)

        # Assert the model is trained and not None
        self.assertIsNotNone(model)

    def test_model_evaluation(self):
        # Create a test DataFrame with predictions
        test_data = [
            (1, "2022-05-04", "comment 1", 1, 0.8),
            (2, "2022-05-04", "comment 2", 1, 0.6),
            (3, "2022-05-05", "comment 3", 0, 0.4),
        ]
        df = self.spark.createDataFrame(test_data, ["ID", "date_stock", "comment", "label", "prediction"])

        # Evaluate the model
        accuracy = evaluate_model(df)

        # Assert the accuracy is within the expected range
        self.assertGreaterEqual(accuracy, 0.4)
        self.assertLessEqual(accuracy, 1.0)


    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName('SentimentAnalyzerTest').getOrCreate()
        cls.spark.sparkContext.setLogLevel("ERROR")

    def test_pipeline(self):
        # Load test data
        test_data = self.spark.createDataFrame([
            ("2022-05-04", 1, "This is a positive comment"),
            ("2022-05-04", 0, "This is a negative comment")
        ], ["date_stock", "label", "comment"])

        # Run the pipeline
        df2 = self.run_pipeline(test_data)

        # Check the transformed DataFrame
        expected_data = [
            ("2022-05-04", 1, "this is a positive comment"),
            ("2022-05-04", 0, "this is a negative comment")
        ]
        expected_df = self.spark.createDataFrame(expected_data, ["date_stock", "label", "comment"])

        self.assertDataFrameEqual(df2, expected_df)

    def run_pipeline(self, df):
        from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, VectorAssembler
        from pyspark.ml import Pipeline

        # Preprocessing
        regexTokenizer = RegexTokenizer(inputCol="comment", outputCol="tokens", pattern="\\W+")
        swr = StopWordsRemover(inputCol="tokens", outputCol="Comments")
        cv = CountVectorizer(inputCol="Comments", outputCol="token_features", minDF=2.0)
        vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")

        # Define the stages for the pipeline
        stages = [regexTokenizer, swr, cv, vecAssembler]

        # Create the pipeline
        pipeline = Pipeline(stages=stages)

        # Fit the pipeline on the input DataFrame
        data = pipeline.fit(df).transform(df)

        return data

    def assertDataFrameEqual(self, df1, df2):
        # Compare the schemas
        self.assertEqual(df1.schema, df2.schema)

        # Compare the data content
        self.assertEqual(df1.collect(), df2.collect())


    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName('SentimentAnalyzerTest').getOrCreate()
        cls.spark.sparkContext.setLogLevel("ERROR")
        cls.data_path = "/path/to/StockPredict_Data.csv"


    def test_data_loading(self):
        df = self.spark.read.format("csv")\
            .option("header", "true")\
            .option("inferSchema", "true")\
            .option("multiLine", "true")\
            .option("delimiter", "¥")\
            .load(self.data_path)\
            .coalesce(5)

        self.assertTrue(df is not None)

    def test_data_preprocessing(self):
        # Assuming df is preloaded
        df = ...

        # Preprocessing steps
        df = df.withColumn('comment', col('comment').lower())

        # Add more preprocessing steps here

        # Test assertions
        self.assertTrue(...)

    def test_bloom_filter(self):
        # Assuming df is preloaded
        df = ...

        # Test bloom filter functionality
        bloomFilterIDS = BloomFilter(df.count(), 0.000000001)
        # Add code to populate bloom filter

        # Test assertions
        self.assertTrue(...)

    def test_ml_pipeline(self):
        # Assuming df2 is preprocessed and ready
        df2 = ...

        # Create ML pipeline stages
        regexTokenizer = RegexTokenizer(inputCol="comment", outputCol="tokens", pattern="\\W+")
        stopWordsRemover = StopWordsRemover(inputCol="tokens", outputCol="comments")
        countVectorizer = CountVectorizer(inputCol="comments", outputCol="token_features", minDF=2.0)
        vectorAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")
        naiveBayes = NaiveBayes(smoothing=1.0, modelType="multinomial")
        pipeline = Pipeline(stages=[regexTokenizer, stopWordsRemover, countVectorizer, vectorAssembler, naiveBayes])

        # Split data into train and test sets
        train, test = df2.randomSplit([0.7, 0.3])

        # Fit the pipeline on the train set
        model = pipeline.fit(train)

        # Make predictions on the test set
        predictions = model.transform(test)

        # Evaluate the model
        evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
        accuracy = evaluator.evaluate(predictions)

        # Test assertions
        self.assertTrue(...)

    def test_cross_validation(self):
        # Assuming df2 is preprocessed and ready
        df2 = ...

        # Create ML pipeline stages
        regexTokenizer = RegexTokenizer(inputCol="comment", outputCol="tokens", pattern="\\W+")
        stopWordsRemover = StopWordsRemover(inputCol="tokens", outputCol="comments")
        countVectorizer = CountVectorizer(inputCol="comments", outputCol="token_features", minDF=2.0)
        vectorAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")
        naiveBayes = NaiveBayes(smoothing=1.0, modelType="multinomial")
        pipeline = Pipeline(stages=[regexTokenizer, stopWordsRemover, countVectorizer, vectorAssembler, naiveBayes])
        
    # Split data into train and test sets
    train, test = df2.randomSplit([0.7, 0.3])

    # Define parameter grid for cross-validation
    paramGrid = ParamGridBuilder().addGrid(naiveBayes.smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]).build()

    # Create cross-validator
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator)

    # Fit cross-validator on the train set
    cvModel = cv.fit(train)

    # Make predictions on the test set
    cvPredictions = cvModel.transform(test)

    # Evaluate the best model from cross-validation
    cvAccuracy = evaluator.evaluate(cvPredictions)

    # Test assertions
    self.assertTrue(...)
    

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName('SentimentAnalyzer').getOrCreate()

    def test_naive_bayes_model(self):
        # Load test data
        test_data = self.spark.read.format("csv").option("header", "true").load("test_data.csv")

        # Prepare test data
        # ... your test data preparation code ...

        # Instantiate and fit Naive Bayes model
        model = NaiveBayes(smoothing=1.0, modelType="multinomial")
        trained_model = model.fit(train_data)

        # Make predictions
        predictions = trained_model.transform(test_data)

        # Assert your test assertions
        # ... your assertions ...

    def test_logistic_regression_model(self):
        # Load test data
        test_data = self.spark.read.format("csv").option("header", "true").load("test_data.csv")

        # Prepare test data
        # ... your test data preparation code ...

        # Instantiate and fit Logistic Regression model
        model = LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
        trained_model = model.fit(train_data)

        # Make predictions
        predictions = trained_model.transform(test_data)

    @classmethod
    def setUpClass(cls):
        # Set up Spark session
        cls.spark = SparkSession.builder.appName('SentimentAnalyzerTest').getOrCreate()


    def setUp(self):
        # Load test data
        self.test_data = self.spark.createDataFrame([
            ("This is a positive comment", 1),
            ("This is a negative comment", 0)
        ], ["comment", "label"])

    def tearDown(self):
        # Clean up any temporary resources
        pass

    def test_preprocessing(self):
        # Test preprocessing step
        preprocessed_data = preprocess_data(self.test_data)
        
        # Assert the output matches the expected results
        expected_data = self.spark.createDataFrame([
            ("this is a positive comment", 1),
            ("this is a negative comment", 0)
        ], ["comment", "label"])
        self.assertEqual(preprocessed_data.collect(), expected_data.collect())

    def test_sentiment_analysis(self):
        # Test sentiment analysis step
        result = sentiment_analysis(self.test_data)
        
        # Assert the output matches the expected results
        expected_result = self.spark.createDataFrame([
            ("This is a positive comment", 1, 1),
            ("This is a negative comment", 0, 0)
        ], ["comment", "label", "prediction"])

        @classmethod
        def tearDownClass(cls):
            cls.spark.stop()
    
if __name__ == '__main__':
        unittest.main()