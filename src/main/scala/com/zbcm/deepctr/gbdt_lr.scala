package com.zbcm.deepctr

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.{BoostingStrategy, FeatureType, Strategy}
import org.apache.spark.mllib.tree.model.{GradientBoostedTreesModel, Node}
import org.apache.spark.rdd._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.{SparkConf, SparkContext}

object gbdt_lr {
  //get decision tree leaf's nodes
  def getLeafNodes(node: Node): Array[Int] = {
    var treeLeafNodes = new Array[Int](0)

    if (node.isLeaf) {
      treeLeafNodes = treeLeafNodes.:+(node.id)   // :+ 方法往数组中追加内容
    } else {
      treeLeafNodes = treeLeafNodes ++ getLeafNodes(node.leftNode.get)
      treeLeafNodes = treeLeafNodes ++ getLeafNodes(node.rightNode.get)
    }
    treeLeafNodes

  }

  // predict decision tree leaf's node value
  def predictModify(node: Node, features: DenseVector): Int = {
    val split = node.split
    if (node.isLeaf) {
      node.id
    } else {
      if (split.get.featureType == FeatureType.Continuous) {  //获取节点类型 Continuous 表示还有子节点 Categorical 表示是叶节点
        if (features(split.get.feature) <= split.get.threshold) {  // split.get.threshold 节点分隔值
          //          println("Continuous left node")
          predictModify(node.leftNode.get, features)
        } else {
          //          println("Continuous right node")
          predictModify(node.rightNode.get, features)
        }
      } else {
        if (split.get.categories.contains(features(split.get.feature))) {
          //          println("Categorical left node")
          predictModify(node.leftNode.get, features)
        } else {
          //          println("Categorical right node")
          predictModify(node.rightNode.get, features)
        }
      }
    }
  }


  def getMetrics(model: GradientBoostedTreesModel,data: RDD[LabeledPoint]): BinaryClassificationMetrics={
    val predictionAndLabels= data.map(example =>
      (model.predict(example.features),example.label)
    )
    new BinaryClassificationMetrics(predictionAndLabels)
  }


  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val sparkConf = new SparkConf().setAppName("GbdtAndLr").setMaster("local")
    sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    //val sampleDir = "/Users/leiyang/IdeaProjects/spark_2.3/src/watermelon3_0_En.csv"
    val sampleDir="D:/douyinData/smallTrain.csv"
    //val sampleDir="file:///data/code/DeepCTR/data/dataForSkearn/train.csv"
    val sc = new SparkContext(sparkConf)
    val spark = SparkSession.builder.config(sparkConf).getOrCreate()
    val df_origin= spark.read.format("CSV").option("header", "false").load(sampleDir)
    //删除第一行数据

    //val df_origin=df_origin1.where(" _c0 != 'item_id' ")

    df_origin.show(5,truncate = false)
    // item_id 0,uid 1,channel 2,finish 3,like 4,duration_time 5,item_pub_hour 6,device_Cnt_bin 7,authorid_Cnt_bin 8 ,musicid_Cnt_bin 9,uid_playCnt_bin 10,itemid_playCnt_bin 11,user_city_score_bin 12,item_city_score_bin 13,
    // title_topic 14 ,gender 15 ,beauty 16 ,relative_position_0 17 ,relative_position_1 18 ,relative_position_2 19 ,relative_position_3 20
    val colNames = Array("_c2","_c3","_c4", "_c5","_c6","_c7","_c8","_c9","_c10","_c11","_c12","_c13","_c14","_c15","_c16","_c17","_c18","_c19","_c20")
    var df1 = df_origin
    for (colName <- colNames) {
      df1 = df1.withColumn(colName, col(colName).cast(DoubleType))
    }

    df1.show(5,truncate = false)

    val df=df1.na.drop()

    //不同特征值的个数
    println("各类别变量特征值的个数")
    df.agg(countDistinct("_c2").alias("2_distinct"),
      countDistinct("_c6").alias("6_distinct"),
      countDistinct("_c7").alias("7_distinct"),
      countDistinct("_c8").alias("8_distinct"),
      countDistinct("_c9").alias("9_distinct"),
      countDistinct("_c10").alias("10_distinct"),
      countDistinct("_c11").alias("11_distinct"),
      countDistinct("_c12").alias("12_distinct"),
      countDistinct("_c13").alias("13_distinct"),
      countDistinct("_c14").alias("14_distinct")).show()

    // Map feature names to indices
    val featInd = List("_c2", "_c5","_c6","_c7","_c8","_c9","_c10","_c11","_c12","_c13","_c14","_c15","_c16","_c17","_c18","_c19","_c20").map(df.columns.indexOf(_))
    println(featInd)
    // Get index of target  _c4:表示like
    val targetInd = df.columns.indexOf("_c4")
    println(targetInd)

    val data = df.rdd.map(r => LabeledPoint(
      r.getDouble(targetInd),
      new DenseVector(featInd.map(r.getDouble(_)).toArray)
    ))

    //data.collect().foreach(println(_))

   //准确率低的原因：1、只用了一小部分数据进行训练；2、只用一小部分字段进行训练
    //解决办法：1、用全部数据进行训练  2、用全部字段进行训练  3、调参
    val splits = data.randomSplit(Array(0.8, 0.2))
    val train = splits(0)
    val test = splits(1)

    // GBDT Model
    val numTrees = 100
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.setNumIterations(numTrees)
    boostingStrategy.learningRate = 0.1
    val treeStratery = Strategy.defaultStrategy("Classification")

   // val categoricalFeaturesInfo = Map[Int, Int]((2,5),(6,12),(7,7),(8,6),(9,7),(10,7),(11,5),(12,4),(13,4),(14,51))    //序号及其特征值个数
    treeStratery.setMaxDepth(4)
    treeStratery.setNumClasses(2)
   // treeStratery.setCategoricalFeaturesInfo(categoricalFeaturesInfo)   //会报错，所以注释掉了，跟上面的Map有关

    //treeStratery.setCategoricalFeaturesInfo(Map[Int, Int]())
    boostingStrategy.setTreeStrategy(treeStratery)

    val gbdtModel = GradientBoostedTrees.train(train, boostingStrategy)
    //    val gbdtModelDir = args(2)
    //    gbdtModel.save(sc, gbdtModelDir)
    val labelAndPreds = test.map { point =>
      val prediction = gbdtModel.predict(point.features)
      (point.label, prediction)
    }
    //val metrics_g =getMetrics(gbdtModel,test)
    //在这一步衡量分类器的好坏，二分类的auc值
    val metrics_gbdt = new BinaryClassificationMetrics(labelAndPreds)
    val auc_gbdt = metrics_gbdt.areaUnderROC()
    println(s"Area under ROC = $auc_gbdt")    //但是auc只有0.5多


    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / test.count()
    println("Test Error = " + testErr)     //这边testErr基本是0.011

    //    println("Learned classification GBT model:\n" + gbdtModel.toDebugString)

    val treeLeafArray = new Array[Array[Int]](numTrees)
    for (i <- 0.until(numTrees)) {
      //println("正在打印第%d 棵树", i)
      treeLeafArray(i) = getLeafNodes(gbdtModel.trees(i).topNode)
    }
   /* for (i <- 0.until(numTrees)) {
      println("正在打印第%d 棵树的 topnode 叶子节点", i)
      print("叶子节点数",treeLeafArray(i).length)
      for (j <- 0.until(treeLeafArray(i).length)) {
        println(j)
      }
    }*/
    // gbdt 构造新特征
    val newFeatureDataSet = df.rdd.map { r => (
      r.getDouble(targetInd),
      new DenseVector(featInd.map(r.getDouble(_)).toArray)
    )} .map { x =>
      var newFeature = new Array[Double](0)
      for (i <- 0.until(numTrees)) {
        val treePredict = predictModify(gbdtModel.trees(i).topNode, x._2)
        //gbdt tree is binary tree
        val treeArray = new Array[Double]((gbdtModel.trees(i).numNodes + 1) / 2)
        treeArray(treeLeafArray(i).indexOf(treePredict)) = 1
        newFeature = newFeature ++ treeArray
      }
      (x._1, newFeature)
    }
    val newData = newFeatureDataSet.map(x => LabeledPoint(x._1, new DenseVector(x._2)))
    val splits2 = newData.randomSplit(Array(0.8, 0.2))
    val train2 = splits2(0)
    val test2 = splits2(1)

    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(train2)    //.setThreshold(0.8)  //设置用于区分正负样本的阈值。当预测值大于该预置时，判定为正样本
    model.weights
    val predictionAndLabels = test2.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val auc = metrics.areaUnderROC()
    println(s"Area under ROC = $auc")

    // Precision by threshold
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }

    // Precision-Recall Curve
    val PRC = metrics.pr

    // F-measure
    val f1Score = metrics.fMeasureByThreshold
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }

    // AUPRC
    val auPRC = metrics.areaUnderPR
    println(s"Area under precision-recall curve = $auPRC")

    // Compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)

    // ROC Curve
    val roc = metrics.roc

    // AUROC
    val auROC = metrics.areaUnderROC
    println(s"Area under ROC = $auROC")


    sc.stop()
  }
}

