from pyspark import SparkContext, SparkConf
import sys
import os
import numpy as np
import time


# ---------------------------------------------- INPUT CHECKING -----------------------------------------------

# Function to check positive integer (Ka, Kb)
def check_positive_int(value, name):
    try:
        n = int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer")

    if n < 0:
        raise ValueError(f"{name} must be greater than or equal to 0")
    return n

# Function to check non negative integer (L)
def check_non_negative_int(value, name):
    try:
        n = int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer")

    if n <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return n

# ---------------------------------------------- OBJECTIVE FUNCTION CALCULATION -----------------------------------------------

# TODO: non bisognerebbe calcolarla su tutti i punti insieme ma passo dopo passo nel map reduce
def calc_objective_function(points, centroids):
    max_dist = 0

    points = np.asarray(points.map(lambda p: p[0]).collect())
    centroids_points = np.asarray([c[0] for c in centroids])

    # Find the maximum of the minimum distances
    for point in points:
        # Distance from this point to ALL centroids
        distances = np.linalg.norm(centroids_points - point, axis=1)

        # Distance to the NEAREST centroid
        min_dist = np.min(distances)

        # Update maximum distance found so far
        max_dist = max(max_dist, min_dist)

    print("Objective function =", max_dist)

    return max_dist

# ---------------------------------------------- MAP REDUCE FAIR FFT -----------------------------------------------

# FFT algorithm for both universes and plot the centroids and the points (if 2D)
def Fair_FFT(X, ka, kb):
    """
    X: iterator of points (x, label) where x is a tuple of coordinates and label is either 'A' or 'B'
    ka: number of centroids of a
    kb: number of centroids of b

    out: centroids of a and b
    """

    # GET THE POINTS FOR EACH UNIVERSE

    data = list(X)

    N = len(data)

    points = []
    labels = []

    # Points split, maybe this and checking ka and kb is optional
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

    # First centroid

    centroids = []
    centroids_labels = []

    # Counters
    centroids_a = 0
    centroids_b = 0
    added = False

    # First centroid

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

    centroids = list(zip(centroids, centroids_labels))

    return centroids


def MRFairFFT(inputPoints, ka, kb, Na, Nb, L):

    beta = 2

    local_ka = int(min(beta*ka/L, (Na / L)))
    local_kb = int(min(beta*kb/L, (Nb / L)))

    coreset = (inputPoints.mapPartitions(lambda it: Fair_FFT(it, local_ka, local_kb))       # 1st ROUND REDUCE, FFT on each partition
                            .coalesce(1)    # Collect to 1 partition
                            .mapPartitions(lambda it: Fair_FFT(it, ka, kb)))     # 2nd ROUND REDUCE, FFT on the aggregated centroids found by each partition

    return coreset.collect()
  


def main():
    # SPARK SETUP
    conf = SparkConf().setAppName('G48HW1')
    sc = SparkContext(conf=conf)

    # Check number of arguments
    if len(sys.argv) != 5:
        print("Usage: G48HW1 <data_path> <Ka> <Kb> <L>")
        return 1

    # Input reading and checking
    data_path = sys.argv[1]

    if not os.path.isfile(data_path):
        print("File not found")
        return 1

    try:
        ka = check_positive_int(sys.argv[2], "ka")
        kb = check_positive_int(sys.argv[3], "kb")
        L = check_non_negative_int(sys.argv[4], "L")
    except ValueError as e:
        print(e)
        return 1

    print('File path: ' + data_path + ' KA: ' + str(ka) + ' KB: ' + str(kb) + " L: " + str(L))

    # Read input file and divide it into L random partitions and divide into tuples of points (x, y, label)
    input_points = (sc.textFile(data_path)
                   .repartition(numPartitions=L)
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

    # Merge all centroids and all points together
    calc_objective_function(input_points, coreset)

    print(f"Running time of MRFairFFT = {duration_ms} ms")

    return 0

if __name__ == "__main__":
    main()