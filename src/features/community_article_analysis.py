# repeat community analysis on simpler graph (all nodes relabeled)
from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.getOrCreate()
assign_file = 'src/data/processed/article_assign5.txt'
remap_file = 'src/data/processed/article_remap.txt'
assigns = spark.read.option("delimiter"," ").csv(assign_file)
remaps = spark.read.option("delimiter","\t").csv(remap_file)
assigns.registerTempTable("assigns")
remaps.registerTempTable("remaps")
maps = spark.sql("""
SELECT article,
CAST(LEFT(article, 1) AS integer) AS quarter,
CAST(concat('200',RIGHT(LEFT(article, 2),1)) AS integer) AS year,
CAST(RIGHT(article, LENGTH(article) - 2) AS integer) AS article_id, community_id
FROM (
SELECT A._c0 AS article, B._c1 AS community_id
FROM remaps A, assigns B
WHERE A._c1 = B._c0
) AS A
""")
maps.registerTempTable("maps")

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

article_contribs = spark.sql("""
SELECT article_id, year, quarter,
SUM(yval) AS sum_yval, COUNT(*) AS num
FROM (
SELECT A.user_id, A.year, A.quarter, B.article_id, yval
FROM contrib A, bipartite B
WHERE A.user_id = B.user_id AND A.quarter = quarter(B.edit_date) AND A.year = year(B.edit_date)
)
GROUP BY article_id, year, quarter
""")
article_contribs.registerTempTable("article_contribs")

# finally get the average for users
# also include just the articles-as-self-community info
user_contribs = spark.sql("""
SELECT user_id, avg_yval, avg_csize, avg_syval, avg_scsize,
FROM (
SELECT user_id, avg_yval, avg_csize, avg_syval, avg_scsize,
rank() OVER (PARTITION BY user_id ORDER BY year, quarter) AS rank
FROM (
SELECT user_id, year, quarter, AVG(avg_yval) AS avg_yval, AVG(c_size) AS avg_csize,
AVG(self_yval) AS avg_syval, AVG(self_csize) AS avg_scsize
FROM (
SELECT A.article_id, A.year, A.quarter, SUM(sum_yval) OVER (PARTITION BY community_id) /
SUM(num) OVER (PARTITION BY community_id) avg_yval,
COUNT(1) OVER (PARTITION BY community_id) AS c_size,
sum_yval/num AS self_yval,
num AS self_csize
FROM article_contribs A, maps B
WHERE A.article_id = B.article_id AND A.quarter = B.quarter AND A.year = B.year
) AS A, bipartite B
WHERE A.article_id = B.article_id AND A.quarter = quarter(B.edit_date) AND A.year = year(B.edit_date)
GROUP BY user_id, year, quarter
) AS A
) AS A
WHERE rank = 1
""")
output_path = "src/data/processed/cafs"
user_contribs.write.option("sep",",").csv(output_path)
