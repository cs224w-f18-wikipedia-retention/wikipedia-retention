# requires reading from user/article features separately
from pyspark.sql import SparkSession, functions as F
ss = SparkSession.builder.getOrCreate()
user_ff = 'user_features.csv'
article_ff = 'article_features.csv'
uff = ss.read.csv(user_ff)
aff = ss.read.csv(article_ff)
uff.registerTempTable("uf")
aff.registerTempTable("af")
