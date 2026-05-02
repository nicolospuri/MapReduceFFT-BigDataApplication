from pyspark import SparkContext, SparkConf
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import math


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
        raise ValueError(f"{name} must be greater than or equal to 0")
    return n

'''
# Function to plot points and centroids (only for 2D data) and save the plot as a PNG file in the output folder
def plot_points(points, centroids, title, filename):
    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Plotting is only supported for 2D data.")

    plt.figure(figsize=(7, 6))
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=40, alpha=0.7, label='Points')

    if len(centroids) > 0:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=180, marker='X', label='Centroids')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    #plt.gca().set_aspect('equal', adjustable='box')  # To have the same scale on both axes
    plt.tight_layout()
    plt.savefig('output/' + filename, dpi=300, bbox_inches='tight')
    plt.close()
'''

# TODO: non bisognerebbe calcolarla su tutti i punti insieme ma passo dopo passo nel map reduce
def calc_objective_function(points, centroids):
    max_dist = 0

    points = np.asarray(points.map(lambda p: p[0]).collect())
    centroids = np.asarray(centroids.map(lambda p: p[0]).collect())

    # Find the maximum of the minimum distances
    for point in points:
        # Distance from this point to ALL centroids
        distances = np.linalg.norm(centroids - point, axis=1)

        # Distance to the NEAREST centroid
        min_dist = np.min(distances)

        # Update maximum distance found so far
        max_dist = max(max_dist, min_dist)

    print("Objective function =", max_dist)

    return max_dist

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

    # Points split
    a_list = []
    b_list = []

    for point, label in data:
        if label == 'A':
            a_list.append(point)
        else:
            b_list.append(point)

    points_a = np.array(a_list)
    points_b = np.array(b_list)

    Na = len(points_a)
    Nb = len(points_b)

    # Check if ka is greater than 0, this should never happen
    if ka < 0:
        raise ValueError("ka must be between 0 and Na")

    ka = min (ka, Na)  # If ka is greater than Na, we can only select Na centroids

    # Check if kb is greater than 0, this should never happen
    if kb < 0:
        raise ValueError("kb must be greater than 0")

    kb = min (kb, Nb)  # If kb is greater than Nb, we can only select Nb centroids

    # FIRST CENTROID FOR A

    dist_a = np.array([])

    if ka == 0 or Na == 0:
        centroids_a = []
    else:
        # Put the first random centroid for a
        first_idx_a = np.random.randint(Na)
        centroids_a = [points_a[first_idx_a]]
        dist_a = np.linalg.norm(points_a - centroids_a[0],
                                axis=1)  # Euclidean distance of all points from the first centroid of a

    # FIRST CENTROID FOR B

    dist_b = np.array([])

    if kb == 0 or Nb == 0:
        centroids_b = []
    else:
        # Put the first random centroid for b
        first_idx_b = np.random.randint(Nb)
        centroids_b = [points_b[first_idx_b]]
        dist_b = np.linalg.norm(points_b - centroids_b[0],
                                axis=1)  # Euclidean distance of all points from the first centroid of b

    # MAIIN ITERATION OF THE FFT ALGORITHM FOT BOTH UNIVERSES

    max_k = max(ka, kb)

    for i in range(1, max_k):
        # fft for a if there are still centroids to select for a, otherwise skip to b
        if i < ka:
            next_idx_a = np.argmax(dist_a)  # Select the point with the maximum distance from the nearest centroid
            centroids_a.append(points_a[next_idx_a])

            new_dist_a = np.linalg.norm(points_a - centroids_a[-1], axis=1)  # Euclidean distance of all points from the new centroid
            dist_a = np.minimum(dist_a, new_dist_a)  # Update the distance of all points from the nearest centroid

        # fft for a if there are still centroids to select for b, otherwise skip
        if i < kb:
            next_idx_b = np.argmax(dist_b)  # Select the point with the maximum distance from the nearest centroid
            centroids_b.append(points_b[next_idx_b])

            new_dist_b = np.linalg.norm(points_b - centroids_b[-1], axis=1)  # Euclidean distance of all points from the new centroid
            dist_b = np.minimum(dist_b, new_dist_b)  # Update the distance of all points from the nearest centroid

    centroids_a = [(tuple(c), 'A') for c in centroids_a]
    centroids_b = [(tuple(c), 'B') for c in centroids_b]

    return centroids_a + centroids_b


def MRFairFFT(inputPoints, ka, kb, Na, Nb, L):

    local_ka = int(min(2*ka/L, (Na / L)))  # DA CAPIRE!!!!!
    local_kb = int(min(2*kb/L, (Nb / L)))  # DA CAPIRE!!!!!

    coreset = (inputPoints.mapPartitions(lambda it: Fair_FFT(it, local_ka, local_kb))
                            .repartition(1)    # Reduce to 1 partition
                            .mapPartitions(lambda it: Fair_FFT(it, ka, kb)))  # Apply FFT again on the coreset

    print("Coreset: ", coreset.collect())

    return coreset
  


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

    print('File path: ' + data_path + ' Ka: ' + str(ka) + ' Kb: ' + str(kb) + " L: " + str(L))

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

    print('N: ' + str(N) + ' Na: ' + str(Na) + ' Nb: ' + str(Nb))

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

    # OBJECTIVE FUNCTION CALCULATION

    # Merge all centroids and all points together
    calc_objective_function(input_points, coreset)

    print(f"Running time of MRFairFFT = {duration_ms} ms")

if __name__ == "__main__":
    main()