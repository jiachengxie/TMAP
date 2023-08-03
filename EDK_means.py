import copy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

class KDTreeNode:
    def __init__(self, data, left, right, depth):
        self.data = data
        self.left = left
        self.right = right
        self.depth = depth

class KDTree:
    def __init__(self, data):
        self.root = self.build_kdtree(data)
        self.data = data

    def build_kdtree(self, data, depth=0):
        k_shape = data.shape[1]  # 数据维度
        nodes = [(data, None, None, depth)]  # 用队列存储待处理的节点
        root = None

        while nodes:
            node_data, left, right, depth = nodes.pop()

            if len(node_data) == 0:
                continue

            axis = depth % k_shape
            sorted_data = node_data[node_data[:, axis].argsort()]
            median_index = len(sorted_data) // 2
            median_point = sorted_data[median_index]
            left_data = sorted_data[:median_index]
            right_data = sorted_data[median_index + 1:]

            node = KDTreeNode(median_point, None, None, depth)

            if depth == 0:
                root = node

            nodes.append((right_data, None, node, depth + 1))
            nodes.append((left_data, node, None, depth + 1))

        return root

    def count_neighbors(self, query_point, radius, threshold):
        if self.root is None:
            return 0

        count = 0
        stack = [self.root]

        while stack:
            current = stack.pop()

            distance = np.linalg.norm(current.data - query_point)
            if distance <= radius:
                count += 1

            if current.left is not None and distance - radius < current.left.data and self.get_node_count(current.left) >= threshold:
                stack.append(current.left)

            if current.right is not None and distance + radius > current.right.data and self.get_node_count(current.right) >= threshold:
                stack.append(current.right)

        return count

    def search_max_neighbors(self, radius_factor, M, threshold):
        if self.root is None:
            return []

        avg_distance = self.calculate_avg_distance()
        radius = avg_distance * radius_factor

        max_neighbors = []
        for point in self.data:
            count = self.count_neighbors(point, radius, threshold)
            max_neighbors.append((point, count))

        max_neighbors.sort(key=lambda x_: x_[1], reverse=True)
        return [x_[0] for x_ in max_neighbors[:M]]

    def get_node_count(self, node):
        if node is None:
            return 0
        return 1 + self.get_node_count(node.left) + self.get_node_count(node.right)

    def calculate_avg_distance(self):
        data = self.data

        # 计算每对数据点之间的欧几里德距离
        diff = data[:, np.newaxis] - data[np.newaxis, :]
        distances = np.linalg.norm(diff, axis=2)

        # 排除对角线上的距离，避免重复计算
        np.fill_diagonal(distances, 0)

        # 计算距离的平均值
        avg_distance = np.mean(distances)

        return avg_distance

def EDKMeans(n_clusters, data, a, r):
    data = copy.deepcopy(data)
    kdtree = KDTree(copy.deepcopy(data))
    initial_centers = kdtree.search_max_neighbors(a, n_clusters, r)
    kmeans = KMeans(n_clusters=n_clusters, init=np.array(initial_centers), n_init=1, max_iter=100)
    return kmeans.fit(data)