# build basic user features from snapshots. label separately
# ex: user_id ->
from pyspark.sql import SparkSession, functions as F
ss = SparkSession.builder.getOrCreate()
ex_parquet = "part-00000-b9d9476b-cc88-44c4-8b82-f39efb715f54-c000.snappy.parquet"
all_parquet = "*.parquet"
df = ss.read.parquet(all_parquet)
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
")

# augment edits with rank features and no bots
dfr = ss.sql( "\
SELECT * \
,date_trunc('quarter', timestamp) AS t \
,rank() OVER (PARTITION BY article_id ORDER BY timestamp) AS article_rank \
,rank() OVER (PARTITION BY user_id ORDER BY timestamp) AS user_rank \
FROM edits \
WHERE user_id NOT LIKE 'ip:%' AND lower(username) NOT LIKE '%bot%' \
")
dfr.registerTempTable("aug_edits")

# intermediate table of first quarters (need to extend when we look at all users)
user_firsts = ss.sql( "\
SELECT user_id, MIN(t) AS first_quarter \
FROM aug_edits \
GROUP BY user_id \
")
user_firsts.registerTempTable("user_firsts")

# user contribution thresholds
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
SELECT A.user_id, A.t, \
COUNT(1) AS num_edits, \
MIN(timestamp) AS first_edit, \
MAX(timestamp) AS last_edit, \
COUNT(DISTINCT article_id) AS distinct_article, \
SUM(cast(minor AS int)) AS num_minors, \
SUM(textdata) AS sum_textdata, \
log(SUM(textdata)) AS logsum_textdata, \
SUM(log(textdata + 1)) AS sumlog_textdata, \
EXP(AVG(LOG(textdata + 1))) AS geom_textdata, \
EXP(AVG(LOG(article_rank + 1))) AS geom_contrib, \
SUM(CASE WHEN textdata > 1000 THEN 1 ELSE 0 END) AS big_edits, \
SUM(CASE WHEN textdata < 20 THEN 1 ELSE 0 END) AS small_edits \
FROM aug_edits A, user_firsts T \
WHERE A.user_id = T.user_id AND T.first_quarter = A.t \
AND A.user_id NOT LIKE 'ip:%' AND lower(username) NOT LIKE '%bot%' \
GROUP BY A.user_id, A.t \
")
buf.registerTempTable("user_bases")

# aggregate user features and turn into numbers
auf = ss.sql( "\
SELECT user_id, t, num_edits, distinct_article, num_minors, sum_textdata \
,logsum_textdata, sumlog_textdata, geom_textdata, geom_contrib \
,big_edits, small_edits \
,unix_timestamp(t) - 979675971 AS t_offset \
,unix_timestamp(last_edit) - unix_timestamp(first_edit) AS t_interval \
,unix_timestamp(first_edit) - unix_timestamp(t) AS t_offset_first \
,unix_timestamp(t) - unix_timestamp(last_edit) AS t_offset_last \
,ROUND(distinct_article * 1.0/num_edits,4) AS p_distinct \
,ROUND(num_minors * 1.0/num_edits,4) AS p_minors \
,ROUND(big_edits * 1.0/num_edits,4) AS p_big \
,ROUND(small_edits * 1.0/num_edits,4) AS p_small \
FROM user_bases \
")
auf.registerTempTable("user_features")

# join features on labels
feature_vecs = ss.sql( "\
SELECT A.*, \
CASE WHEN COALESCE(B.contrib, 0) > 10 THEN 1 ELSE 0 END AS y \
FROM user_features A \
LEFT JOIN user_contrib B \
ON A.user_id = B.user_id \
")

# done building features, now print to file
feature_vecs.write.format("csv").save("user_features")
