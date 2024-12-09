import math

class Brute(object):
    def __init__(self, data, eps = 1.1):
        self.eps = eps
        self.data = data

        self.neighbors = {}
        self.count = []

        # validate data
        if self.data.size == 0:
            raise ValueError("Data is an empty array")

        if eps < 0 or eps == 0:
            raise ValueError("Epsilon cannot be negative or equal to zero")

    def euclidean_distance(self, x1, x2):
        """ Calculates the l2 distance between two vectors """
        distance = 0
        # Squared distance between each coordinate
        for i in range(len(x1)):
            distance += pow((x1[i] - x2[i]), 2)
        return math.sqrt(distance)

    def brute(self, data, eps):
        for i, loci in enumerate(data):
            ne = []
            count = 0
            for j, locj in enumerate(data):
                if self.euclidean_distance(loci,locj) < eps  and i != j:
                    count = count+1
                    ne.append(j)
            self.neighbors[i] = ne
            self.count.append(count)
        return self.count, self.neighbors
