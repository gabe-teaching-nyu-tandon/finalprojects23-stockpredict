import pytest, pyspark

#tests the data loading functionality of the code

def test_data_loading():
    data_df = pyspark.SparkConf.read.format('csv').option('inferSchema', 'true').option('header', 'true').load('Combined_News_DJIA.csv')
    assert data_df.count() > 0

#Test data schema

def test_data_schema():
    data_df = pyspark.SparkConf.read.format('csv').option('inferSchema', 'true').option('header', 'true').load('Combined_News_DJIA.csv')
    expected_columns = ['Date', 'Label', 'Top1', 'Top2', 'Top3', ...]  # List the expected column names

    # Check if the actual column names match the expected column names
    assert data_df.columns == expected_columns

    # Check the data types of each column
    expected_data_types = ['timestamp', 'integer', 'string', 'string', ...]  # List the expected data types
    actual_data_types = [str(field.dataType) for field in data_df.schema.fields]

    assert actual_data_types == expected_data_types

