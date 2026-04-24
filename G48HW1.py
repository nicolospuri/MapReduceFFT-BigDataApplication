from pyspark import SparkContext, SparkConf
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

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

# Function to plot points and centroids
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

# Farthest First Traversal (FFT) algorithm to select k centroids from the input points
def fft(X, k):
    """
    X: input vectors (n_samples by dimensionality)
    D: distance matrix (n_samples by n_samples)
    k: number of centroids

    out: centroids
    """

    points = np.asarray(X)
    N = len(points)

    if k < 0 or k > N:
        raise ValueError("k must be between 0 and N")

    if k == 0 or N == 0:
        return np.array([])

    first_idx = np.random.randint(N)
    centroids = [points[first_idx]]     # Put the first random centroid

    dist = np.linalg.norm(points - centroids[0], axis=1)   # Euclidean distance of all points from the first centroid

    while len(centroids) < k:
        next_idx = np.argmax(dist)     # Select the point with the maximum distance from the nearest centroid
        centroids.append(points[next_idx])

        new_dist = np.linalg.norm(points - centroids[-1], axis=1)
        dist = np.minimum(dist, new_dist)

    centroids = np.array(centroids)

    return centroids

# Calls the FFT algorithm for both universes and plots the centroids
def fair_fft(Xa, Xb, ka, kb):
    """
    Xa: input points of universe a
    Xb: input points of universe b
    ka: number of centroids of a
    kb: number of centroids of b

    out: centroids of a and b
    """
    try:
        centroids_a = fft(Xa, ka)
        print('Centroids of A = ', centroids_a)
        plot_points(Xa, centroids_a, 'Centroids of A', 'centroids_a.png')
    except ValueError as e:
        raise ValueError(f"Error for A: {e}")

    try:
        centroids_b = fft(Xb, kb)
        print('Centroids of B = ', centroids_b)
        plot_points(Xb, centroids_b, 'Centroids of B', 'centroids_b.png')
    except ValueError as e:
        raise ValueError(f"Error for B: {e}")

    return centroids_a , centroids_b



# def MRFairFFT():



def main():
    # CHECKING NUMBER OF CMD LINE PARAMTERS
    if len(sys.argv) != 5:
        print("Usage: G48HW1 <data_path> <Ka> <Kb> <L>")
        return 1

    # SPARK SETUP
    conf = SparkConf().setAppName('G48HW1')
    sc = SparkContext(conf=conf)

    # INPUT READING

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

    if ka > Na:
        print("Ka must be less than or equal to Na")
        return 1
    if kb > Nb:
        print("Kb must be less than or equal to Nb")
        return 1
    if L > N:
        print("L must be less than or equal to N")
        return 1

    # Test FFT function
    Xa = inputPoints.filter(lambda point: point[2] == "A").map(lambda point: (float(point[0]), float(point[1]))).collect()
    Xb = inputPoints.filter(lambda point: point[2] == "B").map(lambda point: (float(point[0]), float(point[1]))).collect()
    try:
        centroids_a, centroids_b = fair_fft(Xa, Xb, ka, kb)
    except ValueError as e:
        print(e)
        return 1

    # Call to MapReduce


if __name__ == "__main__":
    main()