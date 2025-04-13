import numpy as np
import matplotlib.pyplot as plt


def relief_feature_selection(data, labels, k, num_samples=50, num_neighbors=10):
    """
    Relief-F特征选择算法
    data: 输入数据 (n_samples, n_features)
    labels: 样本标签 (n_samples,)
    k: 需选择的特征数量
    num_samples: 采样样本数
    num_neighbors: 选取的最近邻个数
    返回: 按权重排序的特征索引列表 (降序)
    """
    n_samples, n_features = data.shape
    num_samples = min(num_samples, n_samples)
    weights = np.zeros(n_features)
    labels = labels.flatten()  # 确保标签为1D

    # 随机采样样本
    sample_indices = np.random.choice(n_samples, size=num_samples, replace=False)

    for i in sample_indices:
        current_sample = data[i]
        current_label = labels[i]

        # 找到同类和异类样本
        same_mask = (labels == current_label)
        same_mask[i] = False  # 排除自身
        same_class = data[same_mask]
        diff_class = data[~same_mask]

        if len(same_class) == 0 or len(diff_class) == 0:
            continue

        # 计算到同类样本的距离并选取 k 个最近邻
        same_dists = np.sqrt(np.sum((same_class - current_sample) ** 2, axis=1))
        same_neighbors = same_class[np.argsort(same_dists)[:num_neighbors]]

        # 计算到异类样本的距离并选取 k 个最近邻
        diff_dists = np.sqrt(np.sum((diff_class - current_sample) ** 2, axis=1))
        diff_neighbors = diff_class[np.argsort(diff_dists)[:num_neighbors]]

        # 更新权重
        for neighbor in same_neighbors:
            weights -= (current_sample - neighbor) ** 2 / num_neighbors
        for neighbor in diff_neighbors:
            weights += (current_sample - neighbor) ** 2 / num_neighbors

    # 归一化并排序
    weights /= num_samples
    selected_indices = np.argsort(weights)[::-1][:k]

    return selected_indices
    