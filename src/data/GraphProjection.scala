// spark-submit src/data/GraphProjection.scala
import org.apache.spark.sql.DataFrame

val verbose = false

val df = spark.read.parquet("data/processed/enwiki-meta-compact")
df.show(5)

// TODO: add bot label
def getEdges(enwiki: DataFrame, period: String): DataFrame = {
    enwiki.createOrReplaceTempView("enwiki")
    val query = s"""
    with subset as (
        SELECT
            concat(year, '-', quarter) as period,
            article_id,
            cast(user_id as int) as user_id,
            textdata
        FROM
            enwiki
    )
    -- collect the weighted edge-list
    SELECT
        user_id,
        article_id,
        sum(textdata) as word_count,
        count(*) as num_edits
    FROM
        subset
    WHERE
        user_id is not null AND
        period = '${period}'
    GROUP BY 1, 2
    """
    spark.sql(query)
}

val edges = getEdges(df, "2007-1")
edges.printSchema()

if (verbose) {
    edges.cache()
    println(s"Edges: ${edges.count()}")
    println(s"Users: ${edges.select("user_id").distinct().count()}")
    println(s"Articles: ${edges.select("user_id").distinct().count()}")
}
/*
Edges: 4694179
Users: 354500
Articles: 354500
*/

// project using common neighors
def projectCommonNeighbors(edges: DataFrame, threshold: Integer = 0): DataFrame = {
    edges.createOrReplaceTempView("edges")
    val query = s"""
    with unimodal_projection as (
        SELECT
            t1.user_id as e1,
            t2.user_id as e2,
            count(*) as shared_articles
        FROM edges t1
        JOIN edges t2 ON t1.article_id = t2.article_id
        WHERE t1.user_id < t2.user_id
        GROUP BY 1, 2
    )

    SELECT e1, e2, shared_articles
    FROM unimodal_projection
    WHERE shared_articles > ${threshold}
    """
    spark.sql(query)
}

val user_network = projectCommonNeighbors(edges)
user_network.printSchema()

if (verbose) {
    edges.unpersist()
    user_network.cache()
    println(s"User Network - Edges ${user_network.count()}")
    user_network.selectExpr(
        "min(shared_articles)",
        "max(shared_articles)",
        "approx_percentile(cast(shared_articles as double), array(0.25, 0.5, 0.75, 0.9, 0.95, 0.99)) as pct"
    ).show(5, false)


    val size =
        List(1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000)
        .map(x => (x, user_network.where(s"shared_articles > $x").count()))
}

/*
+--------------------+--------------------+------------------------------+
|min(shared_articles)|max(shared_articles)|pct                           |
+--------------------+--------------------+------------------------------+
|1                   |9702                |[1.0, 1.0, 1.0, 2.0, 3.0, 9.0]|
+--------------------+--------------------+------------------------------+

List[(Int, Long)] = List((1,3779272), (2,1688425), (3,1031248), (4,720445), (5,543160), (10,219787), (20,84834), (50,22346), (100,7559), (200,2369), (500,436), (1000,122))*/
*/

// generate edgelists for testing rough AGM performance
import sys.process._

def createAGMTest(user_network: DataFrame): Unit = {
    user_network.cache()
    val basepath = "data/interim"
    val thresholds = List(1000, 500, 200, 100, 50)
    thresholds.foreach(threshold => {
        val path = s"$basepath/agm_2007Q1_$threshold"
        user_network
            .where(s"shared_articles > $threshold").select("e1", "e2")
            .coalesce(1)
            .write.mode("overwrite").csv(path)

        Seq("sh", "-c", s"cp $path/*.csv $path.csv").!
        s"rm -r $path".!
    })

    println("Run these commands manually")
    val agm_command = "realpath dependencies/snap/examples/agmfit/agmfitmain".!!.trim
    thresholds.foreach(threshold => {
        val n = user_network
            .where(s"shared_articles > $threshold")
            .select("e1").distinct().count()
        val base = "data/interim"
        println(s"$agm_command -i:$base/agm_2007Q1_$threshold.csv -o $base/agm/$threshold -l:")
    })
    user_network.unpersist()
}
//createAGMTest(user_network)


// Add threshold based on common neighbors
// Add threshold based on the min-hash jaccard index

import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

// create rdd edgelist
case class UserNetworkEdge(e1: Long, e2: Long, shared_articles: Long)

def createGraph(user_network: DataFrame): Graph[Long, Double] = {
    val edges = user_network
        .as[UserNetworkEdge]
        .rdd.map(r => Edge[Double](r.e1, r.e2, r.shared_articles.toDouble))
    Graph.fromEdges[Long, Double](edges, 0)
}

val G = createGraph(user_network)
    .partitionBy(PartitionStrategy.RandomVertexCut)
G.cache()
G.vertices.count()


def computeAverageClusteringCoeff(G: Graph[Long, Double]): Double = {
    val nodes = G.vertices.count()
    val totalTriangles = G.degrees.mapValues(deg => (deg*(deg-1)/2))
    val actualTriangles = G.triangleCount().vertices
    val coeff = {
        totalTriangles
        .join(actualTriangles)
        // move normalizing coefficient inside to prevent overflows
        .mapValues { case (total: Int, actual: Int) => actual.toDouble/total/nodes }
        .filter(!_._2.isNaN)
    }
    coeff.map(_._2).reduce(_+_)
}

computeAverageClusteringCoeff(G.subgraph(e => e.attr > 1))

// compute average path length
