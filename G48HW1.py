from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand

# def FairFFT(ka, kb, Ua, Ub):



# def MRFairFFT():





def main():
    # CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 4, "Usage: G48HW1 <data_path> <Ka> <Kb> <L>"

    # SPARK SETUP
    conf = SparkConf().setAppName('G48HW1')
    sc = SparkContext(conf=conf)

    # INPUT READING

    # 1. Read number of partitions
    L = sys.argv[3]
    assert L.isdigit() and int(L) > 0, "L is not an integer"

    # 3. Read number of centers
    Ka = sys.argv[2]
    assert Ka.isdigit() and int(Ka) >= 0, "Ka must be an integer"
    Ka = int(Ka)

    Kb = sys.argv[2]
    assert Kb.isdigit() and int(Kb) >= 0, "Ka must be an integer"
    Kb = int(Kb)

    # 2. Read input file and subdivide it into K random partitions
    data_path = sys.argv[0]
    assert os.path.isfile(data_path), "File or folder not found"

    print('File path: ' + data_path + ' Ka: ' + str(Ka) + ' Kb: ' + str(Kb) + " L: " + str(L))

    inputPoints = (sc.textFile(data_path)
                   .repartition(numPartitions=L)
                   .map(lambda line: line.split(","))
                   .map(lambda point: (point(2), (float(point[0]), float(point[1]))))
                   .groupByKey()
                   .cache())

    # SETTING GLOBAL VARIABLES
    numPoints = inputPoints.count()
    print("N = ", numPoints)

    #    inputPoints.map(lambda point: (point[2], (point[0], point[1]))).groupByKey()

    # Call to MapReduce