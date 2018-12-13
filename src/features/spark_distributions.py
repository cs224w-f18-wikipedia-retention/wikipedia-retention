# build basic user features from snapshots. label separately
# ex: user_id ->
from pyspark.sql import SparkSession, functions as F
ss = SparkSession.builder.getOrCreate()
ex_parquet = "part-00000-b9d9476b-cc88-44c4-8b82-f39efb715f54-c000.snappy.parquet"
all_parquet = "*.parquet"
df = ss.read.parquet(all_parquet)
df.registerTempTable("edits")

aug_edits = ss.sql( "\
SELECT article_id, user_id, textdata, timestamp \
FROM edits \
WHERE user_id NOT LIKE 'ip:%' AND lower(username) NOT LIKE '%bot%' \
")
aug_edits.registerTempTable("aug_edits")

article_counts = ss.sql(" \
SELECT num_edits, COUNT(*) AS articles \
FROM ( \
SELECT article_id, COUNT(*) AS num_edits \
FROM aug_edits \
GROUP BY article_id \
) AS A \
GROUP BY num_edits \
ORDER BY num_edits \
")

user_counts = ss.sql(" \
SELECT num_edits, COUNT(*) AS users \
FROM ( \
SELECT user_id, COUNT(*) AS num_edits \
FROM aug_edits \
GROUP BY user_id \
) AS A \
GROUP BY num_edits \
ORDER BY num_edits \
")

word_counts = ss.sql(" \
SELECT textdata, COUNT(*) AS edit_count \
FROM aug_edits \
GROUP BY textdata \
ORDER BY textdata \
")

user_word_counts = ss.sql("\
SELECT total_words, COUNT(*) AS num_users \
FROM ( \
SELECT user_id, SUM(textdata) AS total_words \
FROM aug_edits \
GROUP BY user_id \
) AS A \
GROUP BY total_words \
ORDER BY total_words \
")

article_counts.coalesce(1).write.format("csv").option("header","true").save("article_counts")
user_counts.coalesce(1).write.format("csv").option("header","true").save("user_counts")
word_counts.coalesce(1).write.format("csv").option("header","true").save("word_counts")
user_word_counts.coalesce(1).write.format("csv").option("header","true").save("user_word_counts")

# calculate days between first/last contribution
account_durations = ss.sql(" \
SELECT ROUND(ival / 3600 / 24) AS days, COUNT(*) AS num_users \
FROM ( \
SELECT user_id \
, unix_timestamp(MAX(timestamp)) - unix_timestamp(MIN(timestamp)) AS ival \
FROM aug_edits \
GROUP BY user_id \
) AS A \
GROUP BY days \
ORDER BY days \
")
account_durations.coalesce(1).write.format("csv").option("header","true").save("account_durations")
