# build basic user features from snapshots. label separately
# ex: user_id ->
from pyspark.sql import SparkSession, functions as F
ss = SparkSession.builder.getOrCreate()
ex_parquet = "part-00000-b9d9476b-cc88-44c4-8b82-f39efb715f54-c000.snappy.parquet"
df = ss.read.parquet(ex_parquet)
df.registerTempTable("edits")

# print out percentile edit # for first authors
p_first_author = ss.sql( "\
SELECT percentile(user_rank, array(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)) AS p \
FROM ( \
SELECT article_id \
,rank() OVER (PARTITION BY article_id ORDER BY timestamp) AS article_rank \
,rank() OVER (PARTITION BY user_id ORDER BY timestamp) AS user_rank \
FROM edits \
WHERE user_id NOT LIKE 'ip:%' AND lower(username) NOT LIKE '%bot%' \
) AS A \
WHERE article_rank = 1 \
").collect()

# augment edits with rank features and no bots
dfr = ss.sql( "\
SELECT * \
,rank() OVER (PARTITION BY article_id ORDER BY timestamp) AS article_rank \
,rank() OVER (PARTITION BY user_id ORDER BY timestamp) AS user_rank \
FROM edits \
WHERE user_id NOT LIKE 'ip:%' AND lower(username) NOT LIKE '%bot%' \
")
dfr.registerTempTable("aug_edits")

# intermediate table of first quarters (need to extend when we look at all users)
user_firsts = ss.sql( "\
SELECT user_id, MIN(date_trunc('quarter', timestamp)) AS first_quarter \
FROM aug_edits \
GROUP BY user_id \
")
user_firsts.registerTempTable("user_firsts")

# user contribut
user_contrib = ss.sql( "\
SELECT E.user_id, SUM(LOG(textdata+1)) AS contrib \
FROM aug_edits E, user_firsts F \
WHERE E.user_id = F.user_id \
AND E.timestamp > F.first_quarter + INTERVAL 93 days AND E.timestamp < F.first_quarter + INTERVAL 186 days \
GROUP BY E.user_id \
")
user_contrib.registerTempTable("user_contrib")

# get basic user features, partitioned by quarter
buf = ss.sql( "\
SELECT A.user_id, date_trunc('quarter', timestamp) AS t, \
COUNT(1) AS num_edits, \
MIN(timestamp) AS first_edit, \
MAX(timestamp) AS last_edit, \
COUNT(DISTINCT article_id) AS distinct_article, \
SUM(cast(minor AS int)) AS num_minors, \
EXP(AVG(LOG(textdata + 1))) AS geom_textdata, \
EXP(AVG(LOG(article_rank + 1))) AS geom_contrib, \
SUM(CASE WHEN textdata > 1000 THEN 1 ELSE 0 END) AS big_edits, \
SUM(CASE WHEN textdata < 20 THEN 1 ELSE 0 END) AS small_edits \
FROM aug_edits A, user_firsts T \
WHERE A.user_id = T.user_id AND T.first_quarter = date_trunc('quarter', timestamp) \
AND A.user_id NOT LIKE 'ip:%' AND lower(username) NOT LIKE '%bot%' \
GROUP BY A.user_id, t ")
buf.registerTempTable("user_bases")

# join basic features on labels
vecs = ss.sql( "\
SELECT A.*, \
CASE WHEN COALESCE(B.contrib, 0) > 10 THEN 1 ELSE 0 END AS y \
FROM user_bases A \
LEFT JOIN user_contrib B \
ON A.user_id = B.user_id \
")
