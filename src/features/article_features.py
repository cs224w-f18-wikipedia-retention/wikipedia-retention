# article-user interaction features
# group by time period
from pyspark.sql import SparkSession, functions as F
ss = SparkSession.builder.getOrCreate()
ex_parquet = "part-00000-b9d9476b-cc88-44c4-8b82-f39efb715f54-c000.snappy.parquet"
all_parquet = "*.parquet"
df = ss.read.parquet(all_parquet)
df.registerTempTable("edits")


# reduce edits to smaller pool
reduced_edits = ss.sql("\
SELECT article_id, user_id, date_trunc('quarter', timestamp) AS t \
,date_trunc('quarter',MIN(timestamp) OVER (PARTITION BY user_id)) AS fq_t \
FROM edits \
WHERE user_id NOT LIKE 'ip:%' AND lower(username) NOT LIKE '%bot%' \
")
reduced_edits.registerTempTable("reduced_edits")

# intermediate table of first quarters (need to extend when we look at all users)
filtered_edits = ss.sql( "\
SELECT article_id, user_id, t \
FROM reduced_edits \
WHERE t = fq_t \
")
filtered_edits.registerTempTable("filtered_edits")

aub = ss.sql( "\
SELECT article_id, date_trunc('quarter', timestamp) AS t \
, COUNT(1) AS num_edits \
, SUM(textdata) AS sum_words \
, SUM(log(textdata+1)) AS sumlog_words \
, log(SUM(textdata)) AS logsum_words \
, AVG(log(textdata+1)) AS avglog_words \
, COUNT(DISTINCT user_id) AS unique_users \
, SUM(CASE WHEN textdata > 200 THEN 1 ELSE 0 END) AS big_edits \
, SUM(CASE WHEN textdata < 20 THEN 1 ELSE 0 END) AS small_edits \
, SUM(CASE WHEN user_id LIKE 'ip:%' THEN 1 ELSE 0 END) AS ip_edits \
, SUM(CASE WHEN lower(username) LIKE '%bot%' THEN 1 ELSE 0 END) AS bot_edits \
, COUNT(1) OVER (PARTITION BY article_id) AS total_num_edits \
FROM edits \
GROUP BY article_id, t \
")
aub.registerTempTable("article_features")

# augment article features
aub_aug = ss.sql( "\
SELECT *, num_edits * 1.0 / unique_users AS edits_per_user \
, log(num_edits) AS lognum_edits \
, sumlog_words * 1.0 / unique_users AS avg_user_threshold \
, big_edits * 1.0 / num_edits AS p_big_edits \
, small_edits * 1.0 / num_edits AS p_small_edits \
, ip_edits * 1.0 / num_edits AS p_ip_edits \
, bot_edits * 1.0 / num_edits AS p_bot_edits \
, num_edits * 1.0 / total_num_edits AS p_period_edits \
FROM article_features \
")
aub_aug.registerTempTable("article_aug_features")

# average article features over user interactions
article_vecs = ss.sql( "\
SELECT E.user_id \
,AVG(num_edits) AS art_edits \
,AVG(lognum_edits) AS art_logedits \
,AVG(sum_words) AS art_sumwords \
,AVG(sumlog_words) AS art_sumlogwords \
,AVG(avglog_words) AS art_avglogwords \
,AVG(unique_users) AS art_unique_users \
,AVG(big_edits) AS art_big_edits \
,AVG(small_edits) AS art_small_edits \
,AVG(ip_edits) AS art_ip_edits \
,AVG(bot_edits) AS art_bot_edits \
,AVG(total_num_edits) AS art_total_edits \
,AVG(edits_per_user) AS art_edits_per_user \
,AVG(lognum_edits * avg_user_threshold) / SUM(lognum_edits) AS art_user_threshold \
,AVG(p_big_edits) AS art_p_big_edits \
,AVG(p_small_edits) AS art_p_small_edits \
,AVG(p_ip_edits) AS art_p_ip_edits \
,AVG(p_bot_edits) AS art_p_bot_edits \
,AVG(p_period_edits) AS art_p_period_edits \
FROM filtered_edits E, article_aug_features A \
WHERE E.article_id = A.article_id AND E.t = A.t \
GROUP BY E.user_id \
")

article_vecs.write.format("csv").save("article_features")
