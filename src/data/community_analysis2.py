# repeat community analysis on simpler graph (all nodes relabeled)
from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.getOrCreate()
base_file = 'src/gen-louvain/default_assign.txt'
ind_file = 'src/gen-louvain/ind_assign.txt'
base_df = spark.read.option("delimiter"," ").csv(base_file)
ind_df = spark.read.option("delimiter"," ").csv(ind_file)
base_df.registerTempTable("bases")
ind_df.registerTempTable("inds")
sizes = spark.sql("""
SELECT _c1 AS community, cat,
COUNT(*) AS edges
FROM (
SELECT *, 'bases' AS cat
FROM bases
UNION ALL
SELECT *, 'inds' AS cat
FROM inds
) AS A
GROUP BY _c1, cat
ORDER BY edges DESC
""")
sc = sizes.collect()
sizes.repartition(1).write.option("sep",",").csv("src/data/processed/community_stats")

assign_file = 'src/gen-louvain/graph_labels.txt'
assign_df = spark.read.option("delimiter"," ").csv(assign_file)
assign_df.registerTempTable("assigns")
map_df = spark.sql("""
SELECT user,
CAST(LEFT(user, 1) AS integer) AS quarter,
CAST(concat('200',RIGHT(LEFT(user, 2),1)) AS integer) AS year,
CAST(RIGHT(user, LENGTH(user) - 2) AS integer) AS user_id, community_id
FROM (
SELECT CAST(A._c0 AS string) AS user, B._c1 AS community_id
FROM assigns A, inds B
WHERE A._c1 = B._c0
) AS A
""")
map_df.registerTempTable("maps")

from src.data.snap_import_user_projection import UnimodalUserProjection
input_path = "src/data/processed/enwiki-meta-compact"
model = UnimodalUserProjection(spark).extract(input_path).transform()

# basic community features (avg contribution level)
contrib = spark.sql("""
SELECT user_id,
year(edit_date) AS year,
quarter(edit_date) AS quarter,
log(SUM(log(word_count+1))+1) AS yval
FROM bipartite
GROUP BY user_id, year, quarter
""")
contrib.registerTempTable("contrib")

user_contribs = spark.sql("""
SELECT user_id, avg_yval, c_size
FROM (
SELECT user_id, year, quarter,
AVG(yval) OVER (PARTITION BY community_id) AS avg_yval,
COUNT(1) OVER (PARTITION BY community_id) AS c_size,
rank() OVER (PARTITION BY user_id ORDER BY year, quarter) AS rank
FROM (
SELECT A.user_id, A.year, A.quarter, community_id, yval
FROM contrib A, maps B
WHERE A.user_id = B.user_id AND A.quarter = B.quarter AND A.year = B.year
) AS A
) AS A
WHERE rank = 1
""")
output_path = "src/data/processed/cfs"
user_contribs.write.option("sep",",").csv(output_path)
