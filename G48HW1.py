from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand

# def FairFFT(ka, kb, Ua, Ub):



# def MRFairFFT():





def main():
    # CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 5, "Usage: G48HW1 <data_path> <Ka> <Kb> <L>"

    # SPARK SETUP
    conf = SparkConf().setAppName('G48HW1')
    sc = SparkContext(conf=conf)

    # INPUT READING

    data_path = sys.argv[1]
    assert os.path.isfile(data_path), "File or folder not found"

    Ka = sys.argv[2]
    assert Ka.isdigit() and int(Ka) >= 0, "Ka must be an integer"
    Ka = int(Ka)

    Kb = sys.argv[3]
    assert Kb.isdigit() and int(Kb) >= 0, "Ka must be an integer"
    Kb = int(Kb)

    L = sys.argv[4]
    assert L.isdigit() and int(L) > 0, "L is not an integer"
    L = int(L)

    print('File path: ' + data_path + ' Ka: ' + str(Ka) + ' Kb: ' + str(Kb) + " L: " + str(L))

    # Read input file and subdivide it into L random partitions and divide into tuples of points (x, y, label)
    inputPoints = (sc.textFile(data_path)
                   .repartition(numPartitions=L)
                   .map(lambda line: line.split(","))
                   .map(lambda point: (float(point[0]), float(point[1]), point[2]))
                   .cache())

    # Counting number of points in the input file and number of points with label A and B
    N = inputPoints.count()
    print('N = ', N)

    Na = inputPoints.filter(lambda point: point[2] == "A").count()
    Nb = N - Na

    print('Na = ', Na, ' Nb = ', Nb)

    assert Ka <= Na, "Ka must be less than or equal to Na"
    assert Kb <= Nb, "Kb must be less than or equal to Nb"
    assert L <= N, "L must be less than or equal to N"

    # Call to MapReduce


if __name__ == "__main__":
    main()