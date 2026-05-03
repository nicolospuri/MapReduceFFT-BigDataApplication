import math

from pyspark import SparkContext, SparkConf
import sys
import os
import numpy as np
import time


# ---------------------------------------------- INPUT CHECKING -----------------------------------------------

def check_arguments(args):
    # Check number of arguments
    if len(sys.argv) != 5:
        raise ValueError("Usage: G48HW1 <data_path> <Ka> <Kb> <L>")

    data_path = sys.argv[1]
    if not os.path.isfile(data_path):
        raise ValueError("File not found")

    try:
        ka = int(sys.argv[2])
    except ValueError:
        raise ValueError("Ka must be an integer")
    if ka < 0:
        raise ValueError("Ka must be greater than or equal to 0")

    try:
        kb = int(sys.argv[3])
    except ValueError:
        raise ValueError("Kb must be an integer")
    if kb < 0:
        raise ValueError("Kb must be greater than or equal to 0")

    try:
        L = int(sys.argv[4])
    except ValueError:
        raise ValueError("L must be an integer")
    if L < 0:
        raise ValueError("L must be greater than or equal to 0")

    return data_path, ka, kb, L

# ---------------------------------------------- OBJECTIVE FUNCTION CALCULATION -----------------------------------------------

def calc_objective_function(points, centroids):
    centroids_points = np.asarray([c[0] for c in centroids])
    max_dist = points.map(lambda p: np.linalg.norm(p[0] - centroids_points, axis=1).min()).max()

    return max_dist

# ---------------------------------------------- MAP REDUCE FAIR FFT -----------------------------------------------

# FFT algorithm for both universes and plot the centroids and the points (if 2D)
def Fair_FFT(X, ka, kb):
    # -------------------- INPUT CHECKING ---------------------

    data = list(X)

    N = len(data)

    points = []
    labels = []

    # Points split
    a_list = []
    b_list = []

    for point, label in data:
        points.append(point)
        labels.append(label)
        if label == 'A':
            a_list.append(point)
        else:
            b_list.append(point)

    points = np.array(points)
    labels = np.array(labels)

    Na = len(a_list)
    Nb = len(b_list)

    # Check if ka is greater than 0, this should never happen
    if ka < 0:
        raise ValueError("ka must be between 0 and Na")

    ka = min (ka, Na)  # If ka is greater than Na, we can only select Na centroids

    # Check if kb is greater than 0, this should never happen
    if kb < 0:
        raise ValueError("kb must be greater than 0")

    kb = min (kb, Nb)  # If kb is greater than Nb, we can only select Nb centroids

    if N == 0 or ka + kb == 0:
        return []

    # -------------------- FAIR FFT ALGORITHM ----------------------

    centroids = []
    centroids_labels = []

    # Counters
    centroids_a = 0
    centroids_b = 0
    added = False

    # FIRST CENTROID

    while not added:
        first_idx = np.random.randint(N)
        if labels[first_idx] == 'A' and ka > 0:
            centroids.append(points[first_idx])
            centroids_labels.append(labels[first_idx])
            centroids_a += 1
            added = True
        elif labels[first_idx] == 'B' and kb > 0:
            centroids = [points[first_idx]]
            centroids_labels = [labels[first_idx]]
            centroids_b += 1
            added = True

    added = False

    dist = np.linalg.norm(points - centroids[0], axis=1)  # Euclidean distance of all points from the first centroid, axis=2 to operate with rows

    # MAIN LOOP

    for i in range(1, ka+kb):
        while not added:
            next_idx = np.argmax(dist)  # Select the point with the maximum distance from the nearest centroid

            if labels[next_idx] == 'A':
                if centroids_a < ka:
                    centroids.append(points[next_idx])
                    centroids_labels.append(labels[next_idx])
                    centroids_a += 1
                    added = True
                    continue
            else:
                if centroids_b < kb:
                    centroids.append(points[next_idx])
                    centroids_labels.append(labels[next_idx])
                    centroids_b += 1
                    added = True
                    continue

            # Remove current point with max distance and find the next one
            dist = np.delete(dist, next_idx, axis=0)    # axis=0 to delete a row
            points = np.delete(points, next_idx, axis=0)
            labels = np.delete(labels, next_idx, axis=0)

        new_dist = np.linalg.norm(points - centroids[-1], axis=1)  # Euclidean distance of all points from the new centroid
        dist = np.minimum(dist, new_dist)  # Update the distance of all points from
        added = False

    centroids = list(zip(centroids, centroids_labels))      # Create a list of tuples (centroid, label) to return

    return centroids


def MRFairFFT(inputPoints, ka, kb, Na, Nb, L):
    beta = 2

    local_ka = int(min(math.ceil(beta*ka/L), math.ceil(Na / L)))
    local_kb = int(min(math.ceil(beta*kb/L), math.ceil(Nb / L)))

    coreset = (inputPoints.mapPartitions(lambda it: Fair_FFT(it, local_ka, local_kb))       # 1st ROUND REDUCE, FFT on each partition
                            .coalesce(1)    # Collect to 1 partition
                            .mapPartitions(lambda it: Fair_FFT(it, ka, kb))     # 2nd ROUND REDUCE, FFT on the aggregated centroids found by each partition
                            .collect())    # Collect the final centroids to the driver

    return coreset
  


def main():
    # SPARK SETUP
    conf = SparkConf().setAppName('G48HW1')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # CHECKING CMD LINE ARGUMENTS
    try:
        data_path, ka, kb, L = check_arguments(sys.argv)
    except ValueError as e:
        print(e)
        return 1
    print('File path: ' + data_path + ' KA: ' + str(ka) + ' KB: ' + str(kb) + " L: " + str(L))

    # Read input file and divide it into L random partitions and divide into tuples of points (x, y, label)
    input_points = (sc.textFile(data_path)
                   .repartition(numPartitions=L)        # 1st ROUND MAP
                   .map(lambda line: line.split(","))
                   .map(lambda point: (tuple(float(x) for x in point[:-1]), point[-1]))
                   .cache())

    # Counting number of points in the input file and number of points with label A and B
    N = input_points.count()
    Na = input_points.filter(lambda point: point[1] == "A").count()
    Nb = N - Na
    print('N: ' + str(N) + ' NA: ' + str(Na) + ' NB: ' + str(Nb))

    # Checking if Ka, Kb and L are valid
    if ka > Na:
        print("Ka must be less than or equal to Na")
        return 1
    if kb > Nb:
        print("Kb must be less than or equal to Nb")
        return 1
    if L > N:
        print("L must be less than or equal to N")
        return 1
    
    start_time = time.time()

    coreset = MRFairFFT(input_points, ka, kb, Na, Nb, L)

    end_time = time.time()

    duration_ms = int((end_time - start_time) * 1000)

    for i in range(len(coreset)):
        print(f"Center: {coreset[i][0]}, Label: {coreset[i][1]}")

    # OBJECTIVE FUNCTION CALCULATION
    max_dist = calc_objective_function(input_points, coreset)

    print("Objective function =", max_dist)

    print(f"Running time of MRFairFFT = {duration_ms} ms")

    return 0

if __name__ == "__main__":
    main()