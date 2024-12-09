'''
VDBSCAN is a normal DBSCAN algorithm plus a check on a precomputed orientation
vector associated with each point, in order to better cluster molecular dynamics
simulation data. The vector check neighlist_method is based on the paper by Tuukka Verho
et al. :

"Crystal Growth in Polyethylene by Molecular Dynamics: The Crystal Edge and Lame
llar Thickness" Macromolecules 51, 13, 2018
https://doi.org/10.1021/acs.macromol.8b00857

DBSCAN algorithim is based on Ryan Davidson's medium article,
as well as Erik Lindernoren's github library.

This neighlist_method precomputes the neighbor list making it compatible with the balltree
neighlist_method of Jake VanderPlas. BallTree is significantly faster than the brute
neighlist_method

See for more info:

Ryan Davidson
https://scrunts23.medium.com/dbscan-algorithm-from-scratch-in-python-475b82e0571c

Erik Lindernoren
https://github.com/eriklindernoren/ML-From-Scratch

Jake VanderPlas
https://gist.github.com/jakevdp/5216193

Tuukka Verho
https://doi.org/10.1021/acs.macromol.8b00857
'''


class VDBSCAN(object):
    def __init__(self, data, chords, ncpus, cluster_method = 'P2', neighlist_method='BallTree', eps=1.1, lam=0.9, leaf_size=30, min_pts=8, crystalmin = 100):

        # validate data
        if data.size == 0:
            print("Data is an empty array")
            exit()
        self.data = data
        self.labels = [0]*len(data)
        self.cores = [True]*len(data)
        self.npus = ncpus

        if chords is not None:
            if chords.size == 0:
                print("Chord data is an empty array")
                exit()
            else:
                self.chords = chords

        else:
            # if none make an array of all 1s for test cases
            self.chords = [1]*len(data)

        if isinstance(ncpus, int):
            self.ncpus = ncpus
        else:
            print(
                    "ncpus is not an integer")
            exit()

        if cluster_method == 'P2' or cluster_method == 'cosine':
            self.cluster_method = cluster_method
        else:
            print('Unsupported cluster_method type')
            exit()

        if neighlist_method == 'BallTree' or neighlist_method == 'KDTree' or neighlist_method == 'Brute':
            self.neighlist_method = neighlist_method
        else:
            print('Unsupported neighlist_method type')
            exit()

        if eps < 0 or eps == 0:
            print("Epsilon cannot be negative or equal to zero")
            exit()
        self.eps = eps

        if lam < 0 or lam == 0:
            print("Lambda cannot be negative or equal to zero")
            exit()
        self.lam = lam

        if leaf_size < 0 or leaf_size == 0:
            print("Leaf size cannot be negative or equal to zero")
            exit()
        self.leaf_size = leaf_size

        if min_pts < 1:
            print("Minimum samples must be greater than zero")
            exit()
        self.min_pts = min_pts

        if crystalmin < 1:
            print("Crystal minimum must be greater than zero")
            exit()
        self.crystalmin = crystalmin

    def P2(self, ci, cq):
        from numpy import dot, sqrt
        dd = dot(ci, cq)
        magci = 0.
        magcq = 0.

        for i in ci:
            magci += i**2

        for q in cq:
            magcq += q**2

        magci = sqrt(magci)
        magcq = sqrt(magcq)

        if (magci*magcq) < 1E-3:
            magci = 1.
            magcq = 1.

        cos = dd/(magci*magcq)

        return 3./2.*(cos**2)-1./2.

    def dot(self, ci, cq):
        from numpy import dot, sqrt
        dd = dot(ci, cq)
        magci = 0.
        magcq = 0.

        for i in ci:
            magci += i**2

        for q in cq:
            magcq += q**2

        magci = sqrt(magci)
        magcq = sqrt(magcq)

        if (magci*magcq) < 1E-3:
            magci = 1.
            magcq = 1.

        cos = dd/(magci*magcq)

        return cos

    def expand(self, ind, cluster, count_len):
        self.labels[ind] = cluster

        neighborhood = self.neighbors_dict[ind]
        neighborhood_set = set(neighborhood)

        i = 0
        while i < len(neighborhood):

            query = neighborhood[i]

            # if labeled noise by not being an initial core point, claim
            if self.labels[query] == -1:
                self.labels[query] = cluster

            # if in another cluster or unclaimed, claim it and neighbors
            else:
                # elif self.labels[query] == 0:
                self.labels[query] = cluster
                leaves = self.neighbors_dict[query]

                # if it is a core point as well, add to list to look through
                if self.count[query] >= self.min_pts:

                    for j, leaf in enumerate(leaves):
                        if leaf not in neighborhood_set:
                            neighborhood.append(leaf)
                            neighborhood_set.add(leaf)
            # else:

            i = i+1

    def get_neighbors(self):
        '''
        Modified version of distance formula used in DBSCAN. Looks for distances
        and the alignment of the average chord vector of the segment to count
        as a neighbor

        Both neighlist_methods will return:
            count - total # neighbors passing
            neighbors - array of indexes of neighbors passing

        Does not assign core points
        '''
        from numpy import dot

        if self.neighlist_method == 'BallTree':
            # nearest neighbor run

            # from polyalign.nn_ball_tree import BallTree
            # balltree = BallTree(self.data, self.leaf_size)
            # dists, ints = balltree.query(self.data, self.leaf_size)

            from sklearn.neighbors import BallTree
            balltree = BallTree(self.data, leaf_size=self.leaf_size)
            dists, ints = balltree.query(self.data, self.leaf_size)

            pts_len = len(self.data)
            # chord vector sort
            neighbors_dict = {}
            count = []
            for i in range(pts_len):

                # total neighborhood
                neighborhood = ints[i]
                # neighbors that pass
                neighbors = []
                n = 0

                len_dist = len(dists[i])

                for j in range(len_dist):
                    # distances of neighborhood
                    array = dists[i]

                    # neighbor in question
                    query = neighborhood[j]

                    # check if below length, parallelish, in range, or same place

                    if self.cluster_method == 'cosine':
                        if (i == query):
                            continue
                        elif (array[j] <= self.eps) and (self.dot(self.chords[i], self.chords[query]) >= self.lam):
                            neighbors.append(query)
                            n = n+1
                        else:
                            continue
                    else:
                        if (i == query):
                            continue
                        elif (array[j] <= self.eps) and (self.P2(self.chords[i], self.chords[query]) >= self.lam):
                            neighbors.append(query)
                            n = n+1
                        else:
                            continue

                neighbors_dict[i] = neighbors
                count.append(n)

        elif self.neighlist_method == 'KDTree':
            # nearest neighbor run

            # from polyalign.nn_ball_tree import BallTree
            # balltree = BallTree(self.data, self.leaf_size)
            # dists, ints = balltree.query(self.data, self.leaf_size)

            from sklearn.neighbors import KDTree
            balltree = KDTree(self.data, leaf_size=self.leaf_size)
            dists, ints = balltree.query(self.data, self.leaf_size)

            pts_len = len(self.data)
            # chord vector sort
            neighbors_dict = {}
            count = []
            for i in range(pts_len):

                # total neighborhood
                neighborhood = ints[i]
                # neighbors that pass
                neighbors = []
                n = 0

                len_dist = len(dists[i])

                for j in range(len_dist):
                    # distances of neighborhood
                    array = dists[i]

                    # neighbor in question
                    query = neighborhood[j]

                    # check if below length, parallelish, in range, or same place
                    if self.cluster_method == 'cosine':
                        if (i == query):
                            continue
                        elif (array[j] <= self.eps) and (self.dot(self.chords[i], self.chords[query]) >= self.lam):
                            neighbors.append(query)
                            n = n+1
                        else:
                            continue
                    else:
                        if (i == query):
                            continue
                        elif (array[j] <= self.eps) and (self.P2(self.chords[i], self.chords[query]) >= self.lam):
                            neighbors.append(query)
                            n = n+1
                        else:
                            continue

                neighbors_dict[i] = neighbors
                count.append(n)

        else:
            from polyalign.nn_brute import Brute

            brute = Brute(self.data, self.chords, self.eps, self.lam)
            count, neighbors_dict = brute.brute_total()

        return count, neighbors_dict

    def expand_check(self, i):
        if self.labels[i] == 0:
                self.expand(i, self.cluster, self.count_len)
                self.cluster = self.cluster+1

    def fit(self):
        self.cluster = 2

        # get neighbors from distance and vector
        self.count, self.neighbors_dict = self.get_neighbors()

        # remove leaves
        self.count_len = len(self.count)
        for i in range(self.count_len):

            if self.count[i] < self.min_pts:
                self.labels[i] = -1
                self.cores[i] = False
                # consider it a leaf
                self.neighbors_dict[i] = []

        # expand clusters from core nodes
        from joblib import Parallel, delayed
        from rich.progress import track

        Parallel(n_jobs=self.ncpus, require='sharedmem')(
            delayed(self.expand_check)(i) for i in track(range(self.count_len),  description= "{:<25}".format('Clustering ...')))

        return self.labels, self.cores
