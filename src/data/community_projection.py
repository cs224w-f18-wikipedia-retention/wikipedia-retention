# sparse matrix community representation
from pyspark.sql import SparkSession, functions as F
ss = SparkSession.builder.getOrCreate()
ex_parquet = "part-00000-b9d9476b-cc88-44c4-8b82-f39efb715f54-c000.snappy.parquet"
all_parquet = "*.parquet"
df = ss.read.parquet(all_parquet)
df.registerTempTable("edits")

import snap
import numpy as np
import scipy.sparse as sparse

# counts for table
ss.sql("SELECT COUNT(DISTINCT user_id) FROM edits \
WHERE unix_timestamp(timestamp) > unix_timestamp('2007-01-01 00:00:00') \
AND unix_timestamp(timestamp) < unix_timestamp('2007-04-01 00:00:00')\
").show()

ss.sql("SELECT COUNT(DISTINCT article_id) FROM edits \
WHERE unix_timestamp(timestamp) > unix_timestamp('2007-01-01 00:00:00') \
AND unix_timestamp(timestamp) < unix_timestamp('2007-04-01 00:00:00')\
").show()

ss.sql("SELECT COUNT(DISTINCT(user_id,article_id)) FROM edits \
WHERE unix_timestamp(timestamp) > unix_timestamp('2007-01-01 00:00:00') \
AND unix_timestamp(timestamp) < unix_timestamp('2007-04-01 00:00:00')\
").show()
