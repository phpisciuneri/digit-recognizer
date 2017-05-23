package com.github.phpisciuneri.digitrec

import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler

object Main extends App with LazyLogging {

  val spark = SparkSession.builder()
    .master("local[4]")
    .appName("digit-recognition")
    .getOrCreate()

  val rawTrainingData = spark.read.format("csv")
    .option("header",true)
    .option("inferSchema",true)
    .load(getClass.getResource("/train.csv").getPath)

  val featureColumns = { for (i <- 0 to 783) yield {s"pixel$i"} }.toArray

  val assembler = new VectorAssembler()
    .setInputCols(featureColumns)
    .setOutputCol("features")

  val trainingData = assembler.transform(rawTrainingData).select("label","features")

  val lr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
    .setLabelCol("label")
    .setFeaturesCol("features")

  val lrModel = lr.fit(trainingData)

  val rawTestData = spark.read.format("csv")
    .option("header",true)
    .option("inferSchema",true)
    .load(getClass.getResource("/test.csv").getPath)

  val testData = assembler.transform(rawTestData).select("features")

  lrModel.transform(testData)

  spark.stop()
}