'''
Taken from Jake Van der Plas

https://github.com/jakevdp/pyTree
https://gist.github.com/jakevdp/5216193

'''
import numpy as np


class BallTree(object):
    def __init__(self, data, leaf_size=40):
        self.data = data
        self.leaf_size = leaf_size

        # validate data
        if self.data.size == 0:
            raise ValueError("X is an empty array")

        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")

        self.n_samples = self.data.shape[0]
        self.n_features = self.data.shape[1]

        # determine number of levels in the tree, and from this
        # the number of nodes in the tree.  This results in leaf nodes
        # with numbers of points betweeen leaf_size and 2 * leaf_size
        self.n_levels = 1 + np.log2(max(1, ((self.n_samples - 1)
                                            // self.leaf_size)))
        self.n_nodes = int(2 ** self.n_levels) - 1

        # allocate arrays for storage
        self.idx_array = np.arange(self.n_samples, dtype=int)
        self.node_radius = np.zeros(self.n_nodes, dtype=float)
        self.node_idx_start = np.zeros(self.n_nodes, dtype=int)
        self.node_idx_end = np.zeros(self.n_nodes, dtype=int)
        self.node_is_leaf = np.zeros(self.n_nodes, dtype=int)
        self.node_centroids = np.zeros((self.n_nodes, self.n_features),
                                       dtype=float)

        # Allocate tree-specific data from TreeBase
        self._recursive_build(0, 0, self.n_samples)

    def _recursive_build(self, i_node, idx_start, idx_end):
        # initialize node data
        self.init_node(i_node, idx_start, idx_end)

        if 2 * i_node + 1 >= self.n_nodes:
            self.node_is_leaf[i_node] = True
            if idx_end - idx_start > 2 * self.leaf_size:
                # this shouldn't happen if our memory allocation is correct
                # we'll proactively prevent memory errors, but raise a
                # warning saying we're doing so.
                import warnings
                warnings.warn("Internal: memory layout is flawed: "
                              "not enough nodes allocated")

        elif idx_end - idx_start < 2:
            # again, this shouldn't happen if our memory allocation
            # is correct.  Raise a warning.
            import warnings
            warnings.warn("Internal: memory layout is flawed: "
                          "too many nodes allocated")
            self.node_is_leaf[i_node] = True

        else:
            # split node and recursively construct child nodes.
            self.node_is_leaf[i_node] = False
            n_mid = int((idx_end + idx_start) // 2)
            _partition_indices(self.data, self.idx_array,
                               idx_start, idx_end, n_mid)
            self._recursive_build(2 * i_node + 1, idx_start, n_mid)
            self._recursive_build(2 * i_node + 2, n_mid, idx_end)

    def init_node(self, i_node, idx_start, idx_end):
        # determine Node centroid
        for j in range(self.n_features):
            self.node_centroids[i_node, j] = 0
            for i in range(idx_start, idx_end):
                self.node_centroids[i_node, j] += self.data[self.idx_array[i],
                                                            j]
            self.node_centroids[i_node, j] /= (idx_end - idx_start)

        # determine Node radius
        sq_radius = 0
        for i in range(idx_start, idx_end):
            sq_dist = self.rdist(self.node_centroids, i_node,
                                 self.data, self.idx_array[i])
            sq_radius = max(sq_radius, sq_dist)

        self.node_radius[i_node] = np.sqrt(sq_radius)
        self.node_idx_start[i_node] = idx_start
        self.node_idx_end[i_node] = idx_end

        nbrhd = self.data[self.idx_array[idx_start:idx_end]]

    def rdist(self, X1, i1, X2, i2):
        d = 0
        for k in range(self.n_features):
            tmp = (X1[i1, k] - X2[i2, k])
            d += tmp * tmp
        return d

    def min_rdist(self, i_node, X, j):
        d = self.rdist(self.node_centroids, i_node, X, j)
        return max(0, np.sqrt(d) - self.node_radius[i_node]) ** 2

    def query(self, X, k=1, sort_results=True):
        X = np.asarray(X, dtype=float)

        if X.shape[-1] != self.n_features:
            raise ValueError("query data dimension must "
                             "match training data dimension")

        if self.data.shape[0] < k:
            raise ValueError("k must be less than or equal "
                             "to the number of training points")

        # flatten X, and save original shape information
        Xshape = X.shape
        X = X.reshape((-1, self.data.shape[1]))

        # initialize heap for neighbors
        heap = NeighborsHeap(X.shape[0], k)

        for i in range(X.shape[0]):
            sq_dist_LB = self.min_rdist(0, X, i)
            self._query_recursive(0, X, i, heap, sq_dist_LB)

        distances, indices = heap.get_arrays(sort=sort_results)
        distances = np.sqrt(distances)

        # deflatten results
        return (distances.reshape(Xshape[:-1] + (k,)),
                indices.reshape(Xshape[:-1] + (k,)))

    def _query_recursive(self, i_node, X, i_pt, heap, sq_dist_LB):
        #------------------------------------------------------------
        # Case 1: query point is outside node radius:
        #         trim it from the query
        if sq_dist_LB > heap.largest(i_pt):
            pass

        #------------------------------------------------------------
        # Case 2: this is a leaf node.  Update set of nearby points
        elif self.node_is_leaf[i_node]:
            for i in range(self.node_idx_start[i_node],
                           self.node_idx_end[i_node]):
                dist_pt = self.rdist(self.data, self.idx_array[i], X, i_pt)
                if dist_pt < heap.largest(i_pt):
                    heap.push(i_pt, dist_pt, self.idx_array[i])

        #------------------------------------------------------------
        # Case 3: Node is not a leaf.  Recursively query subnodes
        #         starting with the closest
        else:
            i1 = 2 * i_node + 1
            i2 = i1 + 1
            sq_dist_LB_1 = self.min_rdist(i1, X, i_pt)
            sq_dist_LB_2 = self.min_rdist(i2, X, i_pt)

            # recursively query subnodes
            if sq_dist_LB_1 <= sq_dist_LB_2:
                self._query_recursive(i1, X, i_pt, heap, sq_dist_LB_1)
                self._query_recursive(i2, X, i_pt, heap, sq_dist_LB_2)
            else:
                self._query_recursive(i2, X, i_pt, heap, sq_dist_LB_2)
                self._query_recursive(i1, X, i_pt, heap, sq_dist_LB_1)


def _partition_indices(data, idx_array, idx_start, idx_end, split_index):
    # Find the split dimension
    n_features = data.shape[1]

    split_dim = 0
    max_spread = 0

    for j in range(n_features):
        max_val = -np.inf
        min_val = np.inf
        for i in range(idx_start, idx_end):
            val = data[idx_array[i], j]
            max_val = max(max_val, val)
            min_val = min(min_val, val)
        if max_val - min_val > max_spread:
            max_spread = max_val - min_val
            split_dim = j

    # Partition using the split dimension
    left = idx_start
    right = idx_end - 1

    while True:
        midindex = left
        for i in range(left, right):
            d1 = data[idx_array[i], split_dim]
            d2 = data[idx_array[right], split_dim]
            if d1 < d2:
                tmp = idx_array[i]
                idx_array[i] = idx_array[midindex]
                idx_array[midindex] = tmp
                midindex += 1
        tmp = idx_array[midindex]
        idx_array[midindex] = idx_array[right]
        idx_array[right] = tmp
        if midindex == split_index:
            break
        elif midindex < split_index:
            left = midindex + 1
        else:
            right = midindex - 1


class NeighborsHeap:
    def __init__(self, n_pts, n_nbrs):
        self.distances = np.zeros((n_pts, n_nbrs), dtype=float) + np.inf
        self.indices = np.zeros((n_pts, n_nbrs), dtype=int)

    def get_arrays(self, sort=True):
        if sort:
            i = np.arange(len(self.distances), dtype=int)[:, None]
            j = np.argsort(self.distances, 1)
            return self.distances[i, j], self.indices[i, j]
        else:
            return self.distances, self.indices

    def largest(self, row):
        return self.distances[row, 0]

    def push(self, row, val, i_val):
        size = self.distances.shape[1]

        # check if val should be in heap
        if val > self.distances[row, 0]:
            return

        # insert val at position zero
        self.distances[row, 0] = val
        self.indices[row, 0] = i_val

        #descend the heap, swapping values until the max heap criterion is met
        i = 0
        while True:
            ic1 = 2 * i + 1
            ic2 = ic1 + 1

            if ic1 >= size:
                break
            elif ic2 >= size:
                if self.distances[row, ic1] > val:
                    i_swap = ic1
                else:
                    break
            elif self.distances[row, ic1] >= self.distances[row, ic2]:
                if val < self.distances[row, ic1]:
                    i_swap = ic1
                else:
                    break
            else:
                if val < self.distances[row, ic2]:
                    i_swap = ic2
                else:
                    break

            self.distances[row, i] = self.distances[row, i_swap]
            self.indices[row, i] = self.indices[row, i_swap]

            i = i_swap

        self.distances[row, i] = val
        self.indices[row, i] = i_val
