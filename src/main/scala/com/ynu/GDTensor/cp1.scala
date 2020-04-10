package com.ynu.GDTensor

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import java.util.Date

import breeze.stats.distributions.RandBasis

import scala.io.Source


class cp1(sc:SparkContext) {
}

object cp1{
    def init(shapes: Array[Int], rank: Int, myseed: Int=2019): (DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double]) ={
        val A:DenseMatrix[Double] = DenseMatrix.rand(shapes(0), rank, RandBasis.withSeed(myseed-1).uniform)
        val B:DenseMatrix[Double] = DenseMatrix.rand(shapes(1), rank, RandBasis.withSeed(myseed).uniform)
        val C:DenseMatrix[Double] = DenseMatrix.rand(shapes(2), rank, RandBasis.withSeed(myseed+1).uniform)
        (A, B, C)
    }

    def mapperCovertDataFormat(x: String): (Int, Int, Int, Double) ={
        val arr = x.split(",")
        val (i, j, k) = (arr(0) toInt, arr(1) toInt, arr(2) toInt)
        val v = arr(3).toDouble
        (i, j, k, v)
    }

    def calRMSE(sourceFile: Array[String], facMats: (DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double])): Double = {
        var error = 0.0
        val (facA, facB, facC) = facMats
        for(line <- sourceFile){
            val arr = line.split(",")
            val (i, j, k) = (arr(0) toInt, arr(1) toInt, arr(2) toInt)
            val v = arr(3).toDouble
            error += math.pow(v - sum(facA(i,::) *:* facB(j, ::) *:* facC(k, ::)), 2)
        }
        math.sqrt(error / sourceFile.length)
    }

    def main(args: Array[String]): Unit = {
        val cores = 4
//        val filePath = "data/CP/small_ratings3_train.txt"
        val filePath = "data/CP/gData_dim1000_density1e-05.txt"
//        val shapes = Array(610 , 9724 , 24)
        val shapes = Array(1000, 1000, 1000)
        val rank = 10
        val lr = 0.05
        val tol = 0.0001
        val maxIter = 1000 //0.005, 0.01, 0.015, 0.02, 0.025, 0.03,

        val totalTimeStart = new Date().getTime
        val spark = SparkSession.builder().master("local[4]").getOrCreate()
        val sc = spark.sparkContext
        sc.setLogLevel("WARN")
        val rdd = sc.textFile(filePath).repartition(cores).map(mapperCovertDataFormat).cache()
        val (facA, facB, facC) = init(shapes, rank)
        var (broA, broB, broC) = (sc.broadcast(facA), sc.broadcast(facB), sc.broadcast(facC))
        val broLr = sc.broadcast(lr)
        val rddISlice = rdd.map(x => (x._1, x)).groupByKey().cache()
        val rddJSlice = rdd.map(x => (x._2, x)).groupByKey().cache()
        val rddKSlice = rdd.map(x => (x._3, x)).groupByKey().cache()
        rdd.unpersist()

        val sourceFile = Source.fromFile(filePath)
        val sourceLines = sourceFile.getLines().toArray
        sourceFile.close()
//            println(s"iter: 0, RMSE: ${calRMSE(sourceLines, (facA, facB, facC))}")
        for( t <- 1 to maxIter) {
            val startTime = new Date().getTime
            rddISlice.map(I_slice => {
                val (i, values) = I_slice
                val A: DenseMatrix[Double] = broA.value
                val B: DenseMatrix[Double] = broB.value
                val C: DenseMatrix[Double] = broC.value
                val grades = DenseVector.zeros[Double](A.cols).t
                values.foreach(x => {
                    val (i, j, k, v) = x
                    grades :+= (B(j, ::) *:* C(k, ::)) * (v - sum(A(i, ::) *:* B(j, ::) *:* C(k, ::)))
                })
                A(i, ::) :+= grades * broLr.value
                (i, A(i, ::))
            }).collect() foreach {
                case (i, denseVecT) =>
                    facA(i, ::) := denseVecT
            }

            rddJSlice.map(J_slice => {
                val (j, values) = J_slice
                val A: DenseMatrix[Double] = broA.value
                val B: DenseMatrix[Double] = broB.value
                val C: DenseMatrix[Double] = broC.value
                val grades = DenseVector.zeros[Double](B.cols).t
                values.foreach(x => {
                    val (i, j, k, v) = x
                    grades :+= (A(i, ::) *:* C(k, ::)) * (v - sum(A(i, ::) *:* B(j, ::) *:* C(k, ::)))
                })
                B(j, ::) :+= grades * broLr.value
                (j, B(j, ::))
            }).collect() foreach {
                case (j, denseVecT) =>
                    facB(j, ::) := denseVecT
            }

            rddKSlice.map(K_slice => {
                val (k, values) = K_slice
                val A: DenseMatrix[Double] = broA.value
                val B: DenseMatrix[Double] = broB.value
                val C: DenseMatrix[Double] = broC.value
                val grades = DenseVector.zeros[Double](C.cols).t
                values.foreach(x => {
                    val (i, j, k, v) = x
                    grades :+= (A(i, ::) *:* B(j, ::)) * (v - sum(A(i, ::) *:* B(j, ::) *:* C(k, ::)))
                })
                C(k, ::) :+= grades * broLr.value
                (k, C(k, ::))
            }).collect() foreach {
                case (k, denseVecT) =>
                    facC(k, ::) := denseVecT
            }

            broA.unpersist()
            broB.unpersist()
            broC.unpersist()
            broA = sc.broadcast(facA)
            broB = sc.broadcast(facB)
            broC = sc.broadcast(facC)

            if(t % 10 == 0){
                val e = calRMSE(sourceLines, (facA, facB, facC))
                println(s"iter: $t, RMSE: $e, time: ${(new Date().getTime - startTime)/1000.0}s")
                if(e < tol){
                    println(s"total time: ${(new Date().getTime - totalTimeStart)/1000.0}s")
                    sys.exit(0)
                }
            }
        }
        println(s"total time: ${(new Date().getTime - totalTimeStart)/1000.0}s")

        val sourceFile2 = Source.fromFile("data/CP/small_ratings3_test.txt")
        val sourceLines2 = sourceFile2.getLines().toArray
        sourceFile2.close()
        println(s"Tese set error: ${calRMSE(sourceLines2, (facA, facB, facC))}")
    }
}