package com.zbcm.deepctr

import java.util

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.{Level, Logger}
import com.alibaba.fastjson.{JSON, JSONArray, JSONObject}
import com.google.gson.{JsonObject, JsonParser}
//import scala.util.parsing.json._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{IntegerType, StringType, StructType,MapType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.SQLContext
import java.util
import org.apache.spark.ml.linalg.{Vectors,Vector, VectorUDT,SparseVector}
import scala.collection.mutable
import scala.collection.JavaConversions.mapAsScalaMap
import scala.collection.JavaConversions.mutableMapAsJavaMap
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.{LDA, DistributedLDAModel,LDAModel,LocalLDAModel}


object NlpLda {

  def jsonToMap(jsonObj: JSONObject) = {

    val jsonKey = jsonObj.keySet()
    val iter = jsonKey.iterator()
    val map: mutable.HashMap[String, Int] = new mutable.HashMap[String, Int]()
    while (iter.hasNext) {
      val instance = iter.next()
      val value = jsonObj.get(instance).toString.toInt

      map.put(instance, value)
      //println("===key====：" + instance + "===value===：" + value)
    }
    map
  }


  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getRootLogger().setLevel(Level.ERROR)

    val sparkConf= new SparkConf().setAppName("feature engineering of nlpLDA on spark") .set("spark.ui.showConsoleProgress", "false").setMaster("local")
    val sc = new SparkContext(sparkConf)
    //val pathFile="icmechallenge2019/track2/data/track2_title.txt"
    val pathFile="D:/douyinData/track2_title.txt"

    val rawRdd_nlp = sc.textFile(pathFile)
    //{"item_id": 616993, "title_features": {"224": 1, "3363": 1, "1828": 1, "70": 1, "327": 1, "47": 1, "7011": 1, "340": 1, "3191": 1, "4445": 1, "22463": 1}}
    //scala中读取json字符串
    val rawRdd = rawRdd_nlp.map(x => {  //取出每一条数据，把数据转换成JSONObject类型
      (JSON.parseObject(x).getString("item_id"),
        JSON.parseObject(x).getJSONObject("title_features"))

    })
    rawRdd.take(5).foreach(println(_))

    val vocab_size=rawRdd.flatMap(x => x._2).map(_._1).map(x=>x.toString.toInt).max() +1
    println(vocab_size)

    val itemsWithFeatures = rawRdd.mapValues(vs=>vs.toSeq.sortBy(_._1.toInt))
    itemsWithFeatures.take(20).foreach(println(_))

    val sqlContext= new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val df = itemsWithFeatures.map{case (k, v) => (k, Vectors.sparse( vocab_size,v.map(_._1).map(x=>x.toInt).toArray, v.map(_._2).map(x=>x.toString.toDouble).toArray))}.toDF("item_id", "features")
    df.printSchema()
    df.show(5,truncate = false)
    df.cache()

    val lda=new LDA().setK(50).setMaxIter(200)
    val  ldaModel= lda.fit(df)
    //lda模型保存
    ldaModel.write.overwrite().save("D:/douyinData/lda_model_1")
    //lda模型加载
    val  ldaModel_load= LocalLDAModel.load("D:/douyinData/lda_model_1")

    val transformed = ldaModel_load.transform(df)

    transformed.show(2,truncate=false)

    transformed.printSchema()

    val vecToTopicID = udf( (xs: Vector) => xs.toArray.zipWithIndex.maxBy(_._1)._2 )
    val transformedDF = transformed.withColumn("topicID" , vecToTopicID($"topicDistribution") )
    transformedDF.show(2,truncate=false)
    transformedDF.printSchema()


  }
}
