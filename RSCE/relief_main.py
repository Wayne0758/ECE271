import numpy as np
from testSCEPyTorch_ALLAML import load_ALLAML_Data  # 加载数据和分类模块
from FeatureSelectingSCEPyTorch import createOutputAsCentroids  # 引入 centroid 计算函数
from testSCEPyTorch_ALLAML import classifyWithRelief
from sklearn.metrics import accuracy_score

def reliefF_feature_selection(data, labels, k, num_samples=50, num_neighbors=1):
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

def calculate_mse_to_centroids(trData, trLabels, selected_features):
    """
    计算所选特征到类中心（centroid）的均方误差（MSE）距离。
    """
    trData_selected = trData[:, selected_features]
    centroids = createOutputAsCentroids(trData_selected, trLabels)

    mse_total = 0.0
    n_samples = trData_selected.shape[0]

    for i in range(n_samples):
        label = trLabels[i]
        sample = trData_selected[i]
        centroid = centroids[i]  
        mse_total += np.mean((sample - centroid) ** 2)

    mse = mse_total / n_samples
    return mse

if __name__ == "__main__":
    # 超参数设置
    gpuId = 0
    pp = 1  # 数据分区

    # 加载训练和测试数据
    trnSet = load_ALLAML_Data('Trn', pp)
    tstSet = load_ALLAML_Data('Tst', pp)
    trData, trLabels = trnSet[:, :-1], trnSet[:, -1]
    tstData, tstLabels = tstSet[:, :-1], tstSet[:, -1]

    # 特征选择参数
    total_features = trData.shape[1]  
    k_reliefF = 50  # 选取前 50 个特征
    num_neighbors = 10  # 近邻个数

    # 使用 Relief-F 进行特征选择
    reliefF_indices = reliefF_feature_selection(trData, trLabels, k_reliefF, num_neighbors=num_neighbors)
    print(f'Number of Relief-F selected features: {len(reliefF_indices)}')
    print(f"Selected features: {reliefF_indices}")

    # 计算 MSE 到 centroid
    mse = calculate_mse_to_centroids(trData, trLabels, reliefF_indices)
    print(f"MSE distance to centroids: {mse:.6f}")

    # 分类并获取准确率
    fCntList = [k_reliefF]
    accuracyListReliefF = classifyWithRelief(pp, reliefF_indices, fCntList, gpuId)
    accuracy = accuracyListReliefF[0] 

    print(f"Relief-F Accuracy: {accuracy}")
