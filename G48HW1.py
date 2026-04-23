from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
import numpy as np

def Fair_FFT(Xa, Xb, Na, Nb, ka, kb):
    """
    Xa: input points of universe a
    Xb: input points of universe b
    ka: number of centroids of a
    kb: number of centroids of b
    Na: number of points of a
    Nb: number of points of b

    pointsA: np.array of points of universe a
    pointsB: np.array of points of universe b
    distA: list of distances of points of universe a from the nearest centroid
    distB: list of distances of points of universe b from the nearest centroid
    sortedDistA: sorted distA
    sortedDistB: sorted distB

    out: indices of centroids
    """

    # --------------------------------------- Procedure for universe a ------------------------------

    centroidsA = []
    idxA = []
    # Randomly select the first centroid
    ia = np.int32(np.random.uniform(Na))
    pointsA = np.array(Xa)
    distA = []
    sortedDistA = []

    if ka > 0:      # if ka == 0 -> no centroids for universe a, so we skip the selection of centroids for universe a
        centroidsA.append(pointsA[ia])     # Put the first random centroid
        idxA.append(ia)

    if ka > 1:
        # First iteration, just calculate the distances and take the new centroid
        for i in range(0, Na):
            distA.append(np.linalg.norm(centroidsA[0] - pointsA[i]))      # Euclidean distance
            sortedDistA = distA.copy()
        for i in np.argsort(sortedDistA)[::-1]:  # Sort the distances in descending order and select the point with the maximum distance
            centroidsA.append(pointsA[i])
            idxA.append(i)
            break

        # Next iterations, we update the distances and select the new centroid
        while len(centroidsA) < ka:       # While we have not selected enough centroids
            for i in range(0, Na):
                if i not in idxA:     # If the point is not a centroid
                    currDist = np.linalg.norm(centroidsA[-1] - pointsA[i])     # Calculate the distance of the point from the last selected centroid
                    distA.append(min(currDist, distA[i]))      # Update the distance of the point from the nearest centroid
                else:
                    distA[i] = 0      # If the point is a centroid, its distance from the nearest centroid is 0
            sortedDistA = distA.copy()

            for i in np.argsort(sortedDistA)[::-1]:       # Sort the distances in descending order and select the point with the maximum distance
                centroidsA.append(pointsA[i])
                idxA.append(i)
                break

    print('centroidsA = ', centroidsA)

    # ------------------------------ Same procedure for universe b ------------------------------

    centroidsB = []
    idxB = []
    # Randomly select the first centroid
    ib = np.int32(np.random.uniform(Nb))
    pointsB = np.array(Xb)
    distB = []
    sortedDistB = []

    if kb > 0:  # if ka == 0 -> no centroids for universe a, so we skip the selection of centroids for universe a
        centroidsB.append(pointsB[ib])  # Put the first random centroid
        idxB.append(ib)

    if kb > 1:
        # First iteration, just calculate the distances and take the new centroid
        for i in range(0, Nb):
            distB.append(np.linalg.norm(centroidsB[0] - pointsB[i]))  # Euclidean distance
            sortedDistB = distB.copy()
        for i in np.argsort(sortedDistB)[::-1]:  # Sort the distances in descending order and select the point with the maximum distance
            centroidsB.append(pointsB[i])
            idxB.append(i)
            break

        # Next iterations, we update the distances and select the new centroid
        while len(centroidsB) < kb:  # While we have not selected enough centroids
            for i in range(0, Nb):
                if i not in idxB:  # If the point is not a centroid
                    currDist = np.linalg.norm(centroidsB[-1] - pointsB[i])  # Calculate the distance of the point from the last selected centroid
                    distB.append(min(currDist, distB[i]))  # Update the distance of the point from the nearest centroid
                else:
                    distB[i] = 0  # If the point is a centroid, its distance from the nearest centroid is 0
            sortedDistB = distB.copy()

            for i in np.argsort(sortedDistB)[::-1]:  # Sort the distances in descending order and select the point with the maximum distance
                centroidsB.append(pointsB[i])
                idxB.append(i)
                break

    print('centroidsB = ', centroidsB)


    return np.array(centroidsA), np.array(centroidsB)


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

    ka = sys.argv[2]
    assert ka.isdigit() and int(ka) >= 0, "Ka must be an integer"
    ka = int(ka)

    kb = sys.argv[3]
    assert kb.isdigit() and int(kb) >= 0, "Ka must be an integer"
    kb = int(kb)

    L = sys.argv[4]
    assert L.isdigit() and int(L) > 0, "L is not an integer"
    L = int(L)

    print('File path: ' + data_path + ' Ka: ' + str(ka) + ' Kb: ' + str(kb) + " L: " + str(L))

    # Read input file and subdivide it into L random partitions and divide into tuples of points (x, y, label)
    inputPoints = (sc.textFile(data_path)
                   .repartition(numPartitions=L)
                   .map(lambda line: line.split(","))
                   .map(lambda point: (float(point[0]), float(point[1]), point[2])) # TODO: sistemare nel caso di più dimensioni
                   .cache())

    # Counting number of points in the input file and number of points with label A and B
    N = inputPoints.count()
    print('N = ', N)

    Na = inputPoints.filter(lambda point: point[2] == "A").count()
    Nb = N - Na

    print('Na = ', Na, ' Nb = ', Nb)

    assert ka <= Na, "Ka must be less than or equal to Na"
    assert kb <= Nb, "Kb must be less than or equal to Nb"
    assert L <= N, "L must be less than or equal to N"

    # Test FFT function
    Xa = inputPoints.filter(lambda point: point[2] == "A").map(lambda point: (float(point[0]), float(point[1]))).collect()
    Xb = inputPoints.filter(lambda point: point[2] == "B").map(lambda point: (float(point[0]), float(point[1]))).collect()
    centroidsA, centroidsB = Fair_FFT(Xa, Xb, Na, Nb, ka, kb)

    # Call to MapReduce


if __name__ == "__main__":
    main()