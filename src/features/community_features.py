# build basic community features from snapshots.
from pyspark.sql import SparkSession, functions as F
ss = SparkSession.builder.getOrCreate()
ex_parquet = "part-00000-b9d9476b-cc88-44c4-8b82-f39efb715f54-c000.snappy.parquet"
all_parquet = "*.parquet"
df = ss.read.parquet(all_parquet)
df.registerTempTable("edits")

# simplify and filter dataset
dfr = ss.sql( "\
SELECT article_id, user_id, textdata, timestamp, date_trunc('quarter', timestamp) AS t \
FROM edits \
WHERE user_id NOT LIKE 'ip:%' AND lower(username) NOT LIKE '%bot%' \
")
dfr.registerTempTable("aug_edits")

# intermediate table of first quarters
user_firsts = ss.sql( "\
SELECT user_id, MIN(t) AS first_quarter \
FROM aug_edits \
GROUP BY user_id \
")
user_firsts.registerTempTable("user_firsts")

# get user, first quarter, and y value
user_labels = ss.sql( "\
SELECT A.user_id, A.first_quarter, CASE WHEN contrib > 10 THEN 1 ELSE 0 END AS y \
FROM user_firsts A \
LEFT JOIN ( \
SELECT E.user_id, SUM(LOG(textdata+1)) AS contrib \
FROM aug_edits E, user_firsts F \
WHERE E.user_id = F.user_id \
AND E.timestamp > F.first_quarter + INTERVAL 93 days AND E.timestamp < F.first_quarter + INTERVAL 186 days \
GROUP BY E.user_id \
) AS B \
ON A.user_id = B.user_id \
")
user_labels.registerTempTable("user_labels")

# now, reduce aug_edits to only edits in user's first quarter
first_edits = ss.sql( "\
SELECT A.article_id, A.user_id, A.t AS tfq, U.y, COUNT(*) AS num \
FROM aug_edits A, user_labels U \
WHERE A.user_id = U.user_id AND A.t = U.first_quarter \
GROUP BY A.article_id, A.user_id, A.t, U.y \
")
first_edits.registerTempTable("first_edits")

# sanity check
#ss.sql("SELECT COUNT(*), COUNT(DISTINCT(user_id)) FROM first_edits").show()

# sum over interactions for each article
# implicit: ignore cases where n_total = 1 since this will get removed
article_y = ss.sql( "\
SELECT * FROM ( \
SELECT article_id, tfq \
, SUM(y) AS n_pass, COUNT(1) AS n_total \
FROM first_edits \
GROUP BY article_id, tfq \
) AS A \
WHERE n_total > 1 \
")
article_y.registerTempTable("article_y")

# final step: sum over results for each user
user_anorms = ss.sql( "\
SELECT user_id \
, AVG(nu_pass/nu_total) AS anorm_avg \
, SUM(nu_pass)/SUM(nu_total) AS anorm_wavg \
, AVG(LOG(nu_pass/nu_total+1)) AS anorm_logavg \
FROM ( \
SELECT user_id, n_pass - y AS nu_pass, n_total - 1 AS nu_total \
FROM first_edits E, article_y A \
WHERE E.article_id = A.article_id AND E.tfq = A.tfq \
) AS A \
GROUP BY user_id \
")
user_anorms.registerTempTable("user_anorms")

# now join back into user labels
user_article_norms = ss.sql( "\
SELECT A.user_id \
, COALESCE(anorm_avg,0) AS anorm_avg \
, COALESCE(anorm_wavg,0) AS anorm_wavg \
, COALESCE(anorm_logavg,0) AS anorm_logavg \
FROM user_labels A LEFT JOIN user_anorms B \
ON A.user_id = B.user_id \
ORDER BY CAST(A.user_id AS integer) \
")

# test if it worked
#import numpy as np
#res = np.ndarray.astype(np.array(user_article_norms.collect()),float)

user_article_norms.write.format("csv").save("community_norm_features")
