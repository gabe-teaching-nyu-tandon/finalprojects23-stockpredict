import pytest, pyspark
from CountBased import count_based_algo


assert count_based_algo('Combined_News_DJIA.csv') > 0.5

assert count_based_algo('testcase1.csv') > 0.5

assert count_based_algo('testcase2.csv') > 0.5

assert count_based_algo('testcase3.csv') > 0.5

assert count_based_algo('testcase4.csv') > 0.5