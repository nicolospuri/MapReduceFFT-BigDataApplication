centroidsB = []
    # Randomly select the first centroid
    ib = np.int32(np.random.uniform(Nb))
    pointsB = np.array(Xb)
    distB = []
    sortedDistB = []

    if kb > 0:  # if ka == 0 -> no centroids for universe a, so we skip the selection of centroids for universe a
        centroidsB.append(pointsB[ib])  # Put the first random centroid

    if kb > 1:
        # First iteration, just calculate the distances and take the new centroid
        for i in range(0, Nb):
            distB.append(np.linalg.norm(centroidsB[-1] - pointsB[i]))  # Euclidean distance
            sortedDistB = distB.copy()
        for i in np.argsort(sortedDistB)[::-1]:  # Sort the distances in descending order and select the point with the maximum distance
            centroidsB.append(pointsB[i])
            break

        # Next iterations, we update the distances and select the new centroid
        while len(centroidsB) < kb:  # While we have not selected enough centroids
            for i in range(0, Nb):
                if pointsB[i] not in centroidsB:  # If the point is not a centroid
                    currDist = np.linalg.norm(centroidsB[-1] - pointsB[i])  # Calculate the distance of the point from the last selected centroid
                    distB.append(min(currDist, distB[i]))  # Update the distance of the point from the nearest centroid
                else:
                    distB[i] = 0  # If the point is a centroid, its distance from the nearest centroid is 0
            sortedDistB = distB.copy()

            for i in np.argsort(sortedDistB)[::-1]:  # Sort the distances in descending order and select the point with the maximum distance
                if i not in centroidsB:
                    centroidsB.append(pointsB[i])
                    break