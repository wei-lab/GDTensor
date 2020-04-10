package com.ynu.GDTensor

import java.util.Date

import breeze.linalg.{DenseMatrix, sum}
import breeze.stats.distributions.RandBasis
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

import scala.io.Source

class cp3(sc:SparkContext) {
}

object cp3{
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
//        val filePath = "data/CP/gData_dim1000_density1e-05.txt"
//        val shapes = Array(1000, 1000, 1000)
        val filePath = "data/CP/gData_dim1000_density1e-05.txt"
        val shapes = Array(1000, 1000, 1000)
        val rank = 10
//        val lr = 0.01
//        val rho = 10
        val tol = 0.0001
        val maxIter = 1000
//        val totalTimeStart = new Date().getTime
        val spark = SparkSession.builder().master("local[4]").getOrCreate()
        val sc = spark.sparkContext
        sc.setLogLevel("WARN")
        for(lr <- Array(0.05)) {
            for (rho <- 10 to 10) {
                val totalTimeStart = new Date().getTime

                val rdd = sc.textFile(filePath).repartition(cores).map(mapperCovertDataFormat).cache()
                println(rdd.count())
                var (facA, facB, facC) = init(shapes, rank)
                var (broA, broB, broC) = (sc.broadcast(facA), sc.broadcast(facB), sc.broadcast(facC))
                val broLr = sc.broadcast(lr)
                val broRho = sc.broadcast(rho)

                val sourceFile = Source.fromFile(filePath)
                val sourceLines = sourceFile.getLines().toArray
                sourceFile.close()

                var localRDD = rdd.mapPartitions(List(_, ("A", broA.value), ("B", broB.value), ("C", broC.value)).iterator)
                println(s"iter: 0, RMSE: ${calRMSE(sourceLines, (facA, facB, facC))}")

                for (t <- 1 to maxIter) {
                    if (t == 2) rdd.unpersist()
                    val startTime = new Date().getTime

                    // update locally
                    var localOLD = localRDD
                    localRDD = localRDD.mapPartitions(iters => {
                        val lr = broLr.value
                        val alpha = lr * broRho.value
                        val (gA, gB, gC) = (broA.value, broB.value, broC.value)
                        val list = iters.toList
                        val datas = list.head.asInstanceOf[Iterator[(Int, Int, Int, Double)]].toList
                        val facs = list.tail.map(_.asInstanceOf[(String, DenseMatrix[Double])]._2)
                        val (lA, lB, lC) = (facs.head, facs(1), facs(2))
                        for ((i, j, k, v) <- datas) {
                            val error = v - sum(lA(i, ::) *:* lB(j, ::) *:* lC(k, ::))
                            lA(i, ::) :+= (lB(j, ::) *:* lC(k, ::)) * error * lr - (lA(i, ::) - gA(i, ::)) * alpha
                            lB(j, ::) :+= (lA(i, ::) *:* lC(k, ::)) * error * lr - (lB(j, ::) - gB(j, ::)) * alpha
                            lC(k, ::) :+= (lA(i, ::) *:* lB(j, ::)) * error * lr - (lC(k, ::) - gC(k, ::)) * alpha
                        }
                        List(datas.iterator, ("A", lA), ("B", lB), ("C", lC)).iterator
                    }).cache()
                    // update global
                    val beta = lr * rho * cores
                    localRDD.mapPartitions(_.toList.tail.map(_.asInstanceOf[(String, DenseMatrix[Double])]).iterator)
                        .reduceByKey(_ + _).collect() foreach {
                        case (name, factor) =>
                            factor :/= cores.toDouble
                            if (name == "A")
                                facA := facA * (1 - beta) + factor * beta
                            else if (name == "B")
                                facB := facB * (1 - beta) + factor * beta
                            else
                                facC := facC * (1 - beta) + factor * beta
                    }

                    localOLD.unpersist()
                    broA.unpersist()
                    broB.unpersist()
                    broC.unpersist()
                    broA = sc.broadcast(facA)
                    broB = sc.broadcast(facB)
                    broC = sc.broadcast(facC)

                    if(t % 10 == 0) {
                        val e = calRMSE(sourceLines, (facA, facB, facC))
                        println(s"iter: $t, RMSE: $e, time: ${(new Date().getTime - startTime) / 1000.0}s")
                        if (e < tol) {
                            println(s"total time: ${(new Date().getTime - totalTimeStart) / 1000.0}s")
                            sys.exit(0)
                        }
                    }
                }
                println(s"total time: ${(new Date().getTime - totalTimeStart)/1000.0}s")
            }
        }
    }
}