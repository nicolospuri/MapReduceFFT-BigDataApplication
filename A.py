centroidsA = []
    # Randomly select the first centroid
    ia = np.int32(np.random.uniform(Na))
    pointsA = np.array(Xa)
    distA = []
    sortedDistA = []

    if ka > 0:      # if ka == 0 -> no centroids for universe a, so we skip the selection of centroids for universe a
        centroidsA.append(pointsA[ia])     # Put the first random centroid

    if ka > 1:
        # First iteration, just calculate the distances and take the new centroid
        for i in range(0, Na):
            distA.append(np.linalg.norm(centroidsA[0] - pointsA[i]))      # Euclidean distance
            sortedDistA = distA.copy()
        for i in np.argsort(sortedDistA)[::-1]:  # Sort the distances in descending order and select the point with the maximum distance
            centroidsA.append(pointsA[i])
            break

        # Next iterations, we update the distances and select the new centroid
        while len(centroidsA) < ka:       # While we have not selected enough centroids
            for i in range(0, Na):
                if pointsA[i] not in centroidsA:     # If the point is not a centroid
                    currDist = np.linalg.norm(centroidsA[-1] - pointsA[i])     # Calculate the distance of the point from the last selected centroid
                    distA.append(min(currDist, distA[i]))      # Update the distance of the point from the nearest centroid
                else:
                    distA[i] = 0      # If the point is a centroid, its distance from the nearest centroid is 0
            sortedDistA = distA.copy()

            for i in np.argsort(sortedDistA)[::-1]:       # Sort the distances in descending order and select the point with the maximum distance
                if i not in centroidsA:
                    centroidsA.append(pointsA[i])
                    break