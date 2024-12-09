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


class DBSCAN(object):
    def __init__(self, filename, ncpus, neighlist_method='BallTree', eps=1.1, leaf_size=30, min_pts=8, crystalmin = 100):
        from numpy import zeros, array, unique

        if isinstance(filename, str):
            self.filename = filename
        else:
            print("Filename is not a string")
            exit()

        f = open(filename, 'r')
        _data = f.readlines()
        f.close()

        box = _data[5:8]
        _data = _data[9:]

        # atom data
        len_data = len(_data)
        atom = [0]*len_data
        id   = [0]*len_data
        mol  = [0]*len_data
        type = [0]*len_data
        x    = [0]*len_data
        y    = [0]*len_data
        z    = [0]*len_data
        xu   = [0]*len_data
        yu   = [0]*len_data
        zu   = [0]*len_data

        for i in range(len_data):
            idx = _data[i].split()
            atom[i] = float(idx[0])
            id[i] = float(idx[1])
            type[i] = float(idx[2])
            x[i] = float(idx[3])
            y[i] = float(idx[4])
            z[i] = float(idx[5])
            xu[i] = float(idx[6])
            yu[i] = float(idx[7])
            zu[i] = float(idx[8])


        del _data

        atom = array(atom)
        id = array(id)
        moltype = array(type)
        x = array(x)
        y = array(y)
        z = array(z)
        xu = array(xu)
        yu = array(yu)
        zu = array(zu)

        # box data
        xlo, xhi = box[0].split()
        ylo, yhi = box[1].split()
        zlo, zhi = box[2].split()

        xlo = float(xlo)
        xhi = float(xhi)
        ylo = float(ylo)
        yhi = float(yhi)
        zlo = float(zlo)
        zhi = float(zhi)

        data = []
        for i, _ in enumerate(x):
            data.append([x[i], y[i], z[i]])

        data = array(data)

        # validate data
        if data.size == 0:
            print("Data is an empty array")
            exit()
        self.data = data

        self.lims = [[xlo, xhi], [ylo, yhi], [zlo, zhi]]

        self.labels = [0]*len(data)
        self.cores = [True]*len(data)
        self.npus = ncpus

        if isinstance(ncpus, int):
            self.ncpus = ncpus
        else:
            print(
                    "ncpus is not an integer")
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

                    if (i == query):
                        continue
                    elif (array[j] <= self.eps):
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

    def assign(self, labels):
        '''
        Assign cluster labels
        '''

        from numpy import array
        #########################################
        # Filter out based on type
        #########################################
        data = self.data
        _data = []
        for i, _d in enumerate(data):
            _data.append([data[i][0],data[i][1],data[i][2], labels[i]])

        return _data

    def find(self):
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

        data = self.assign(self.labels)

        return data, self.lims
