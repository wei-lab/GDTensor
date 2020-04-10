package com.ynu.GDTensor

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import java.util.Date

import breeze.stats.distributions.RandBasis

import scala.io.{BufferedSource, Source}

class cp2(sc:SparkContext) {
}

object cp2{
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
        val filePath = "data/CP/gData_dim1000_density1e-05.txt"
        val shapes = Array(1000, 1000, 1000)
        val rank = 10
        val lr = 0.05
        val tol = 0.0001
        val maxIter = 1000
        val totalTimeStart = new Date().getTime
        val spark = SparkSession.builder().master("local[4]").getOrCreate()
        val sc = spark.sparkContext
        sc.setLogLevel("WARN")
        val rdd = sc.textFile(filePath).repartition(cores).map(mapperCovertDataFormat).cache()
        println(rdd.count())
        var (facA, facB, facC) = init(shapes, rank)
        var (broA, broB, broC) = (sc.broadcast(facA), sc.broadcast(facB), sc.broadcast(facC))
        val broLr = sc.broadcast(lr)

        val sourceFile = Source.fromFile(filePath)
        val sourceLines = sourceFile.getLines().toArray
        sourceFile.close()
        println(s"iter: 0, RMSE: ${calRMSE(sourceLines, (facA, facB, facC))}")
        for( t <- 1 to maxIter) {
            val startTime = new Date().getTime

            rdd.mapPartitions(iter => {
                val A: DenseMatrix[Double] = broA.value
                val B: DenseMatrix[Double] = broB.value
                val C: DenseMatrix[Double] = broC.value
                val lr = broLr.value
                for((i, j, k, v) <- iter){
                    val error = v - sum(A(i, ::) *:* B(j, ::) *:* C(k, ::))
                    A(i, ::) :+= (B(j, ::) *:* C(k, ::)) * error * lr
                    B(j, ::) :+= (A(i, ::) *:* C(k, ::)) * error * lr
                    C(k, ::) :+= (A(i, ::) *:* B(j, ::)) * error * lr
                }
                List(("A",A), ("B",B), ("C",C)).iterator
            }).reduceByKey( _+_ ).collect() foreach{
                case (name, factor) =>
                    factor := factor / cores.toDouble
                    if(name == "A")
                        facA := factor
                    else if(name == "B")
                        facB := factor
                    else
                        facC := factor
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
    }
}