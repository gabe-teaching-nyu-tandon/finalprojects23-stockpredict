import pytest, pyspark
from CountBased import count_based_algo


assert count_based_algo('Combined_News_DJIA.csv') == 0.505
