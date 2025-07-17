import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
from sklearn.cluster import KMeans
import os
import shutil
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import data_util
from PIL import Image
from config import folder_path, saved_model_path, clusters_amount, subfolder_path

def read_npy_files(folder_path):
    """
    讀取指定資料夾中所有子資料夾中的所有.npy檔案，並返回它們的內容列表和檔名列表。
    
    Args:
    - folder_path (str): 資料夾的路徑，包含子資料夾和.npy檔案。
    
    Returns:
    - file_names (list): 包含所有.npy檔案的檔名的列表。
    - arrays_list (list): 包含所有.npy檔案讀取後的NumPy陣列的列表。
    """
    # 存放檔名和對應陣列的列表
    file_names = []
    arrays_list = []

    # 遍歷資料夾及其子資料夾
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                print(file_path)
                
                # 讀取文件並添加到列表中
                array = np.load(file_path)[:, :, :3]
                arrays_list.append(array)
                # 提取文件名（去掉路徑和擴展名）
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                file_names.append(file_name)

    return file_names, arrays_list


def read_ground_truth(folder_path):
    """
    讀取每個子資料夾中以 GT_label 結尾的 ground truth 檔案，並返回 ground truth 值的列表。
    
    Args:
    - folder_path (str): 資料夾的路徑，包含子資料夾和以 GT_label 結尾的檔案。
    
    Returns:
    - ground_truth_list (list): 包含所有 ground truth 值的列表。
    """
    ground_truth_list = []
    # 遍歷資料夾及其子資料夾
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 檢查文件名是否以 GT_label 結尾
            if file.endswith('GT_label.txt'):
                file_path = os.path.join(root, file)
                # 讀取檔案內容，將每一行的值加入到 ground_truth_list 中
                try:
                    with open(file_path, 'r', encoding="utf-8") as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        gt_value = line.strip()
                        ground_truth_list.append(gt_value)
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='big5') as f:
                        lines = f.readlines()
                    for line in lines:
                        gt_value = line.strip()
                        ground_truth_list.append(gt_value)
    return ground_truth_list


def split_npy_files_by_ground_truth(combined_names, arrays_list, ground_truth_list):
    """
    根據 ground truth 值將 .npy 檔案內容分割成兩個列表：rice_list 和 non_rice_list。
    
    Args:
    - combined_names (list): 包含所有.npy檔案名稱的列表。
    - arrays_list (list): 包含所有.npy檔案內容的列表。
    - ground_truth_list (list): 包含所有 ground truth 值的列表。
    
    Returns:
    - rice_names (list): 包含 ground truth 為 1 的 .npy 檔案名稱的列表。
    - rice_list (list): 包含 ground truth 為 1 的 .npy 檔案內容的列表。
    - non_rice_names (list): 包含 ground truth 為 0 的 .npy 檔案名稱的列表。
    - non_rice_list (list): 包含 ground truth 為 0 的 .npy 檔案內容的列表。
    """
    npy_files = arrays_list
    ground_truths = ground_truth_list

    rice_names = []
    rice_list = []
    non_rice_names = []
    non_rice_list = []

    # 根據 ground truth 值將 .npy 檔案內容分割到不同的列表中
    for combined_name, npy_array, gt_value in zip(combined_names, npy_files, ground_truths):
        if gt_value == 1:
            rice_names.append(combined_name)
            rice_list.append(npy_array)
        elif gt_value == 0:
            non_rice_names.append(combined_name)
            non_rice_list.append(npy_array)

    return rice_names, rice_list, non_rice_names, non_rice_list


def do_Kmeans_clustering(image_list, clusters_amount):
    """
    將每一筆 3 通道 2 維影像攤平成一維陣列，正規化後呼叫 Kmeans 演算法。
    """
    flatten_image_list = np.float32(image_list).reshape(
        len(image_list), -1)  # 將每一筆 3 通道 2 維影像攤平成一維陣列
    flatten_image_list /= 255  # 正規化，縮放至 0~1 之間
    kmeans_model = KMeans(n_clusters=clusters_amount, random_state=0)
    kmeans_model_fit = kmeans_model.fit(flatten_image_list)
    predictions = kmeans_model_fit.labels_  # 分群結果，屬於哪個 cluster

    return predictions


def do_Contrastive_learning(image_list, saved_model_path):
    def _preprocess(x):
        x = data_util.preprocess_image(
            x, 224, 224, is_training=False, color_jitter_strength=0.)
        return x
    
    saved_model = tf.saved_model.load(saved_model_path)
    
    preprocessed_data = list(map(_preprocess, image_list))

    # 創建空列表來儲存預測結果
    outputs = []
    
    for image in preprocessed_data:
        # 對每個預處理後的影像，直接將其傳遞給模型預測
        output = saved_model(image[np.newaxis, ...], False)
        # 將預測結果插入到列表中
        outputs.extend(output.numpy().tolist())
        
    return outputs


def do_tsne(data, clusters, title):
    # 使用 t-SNE 進行降維
    tsne = TSNE(n_components=3, random_state=0)
    outputs_tsne = tsne.fit_transform(data)

    # 可視化原始 CIFAR-10 數據集的 t-SNE 結果
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 設置軸的限制範圍
    ax.set_xlim((-15, 15))
    ax.set_ylim((-15, 15))
    ax.set_zlim((-15, 15))

    # 調整點的大小
    point_size = 10

    # 轉換類別字串為數字
    label_encoder = LabelEncoder()
    c_values = label_encoder.fit_transform(clusters)

    # 繪製 t-SNE 結果，使用不同顏色表示不同的簇
    scatter = ax.scatter(outputs_tsne[:, 0], outputs_tsne[:, 1], outputs_tsne[:, 2], 
                         c=c_values, cmap='viridis', s=point_size)

    # 添加圖例
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    #plt.savefig(f"{saved_model_path}/{title}")
    plt.savefig(f"{subfolder_path}/{title}")
    plt.title(title)
    plt.show()


def calculate_cluster_label_distribution(predictions, ground_truth_labels, num_clusters):
    # 統計每個簇中類別的分佈
    cluster_label_counts = {}

    for cluster_idx in range(num_clusters):
        # 獲取該簇的資料索引
        cluster_indices = np.where(predictions == cluster_idx)[0]
        # 獲取該簇的標籤
        cluster_labels = ground_truth_labels[cluster_indices]
        # 統計該簇中各個類別的個數
        unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
        # 計算佔比
        label_percentages = label_counts / np.sum(label_counts)
        # 將統計結果保存到字典中
        cluster_label_counts[cluster_idx] = {
            'labels': unique_labels,
            'counts': label_counts,
            'percentages': label_percentages
        }

    return cluster_label_counts

def create_distribution_dataframe(cluster_label_counts):
    # 創建一個空的 DataFrame 來存儲類別分佈
    cluster_label_df = pd.DataFrame(columns=['Cluster', 'Class', 'Count', 'Percentage'])

    # 循環每個簇中的類別分佈
    for cluster_idx, counts_info in cluster_label_counts.items():
        print(f"Cluster {cluster_idx} 的類別分佈：")
        # 按佔比大小排序
        sorted_indices = np.argsort(counts_info['percentages'])[::-1]
        sorted_labels = counts_info['labels'][sorted_indices]
        sorted_counts = counts_info['counts'][sorted_indices]
        sorted_percentages = counts_info['percentages'][sorted_indices]
        
        # 將數據添加到 DataFrame 中
        for label, count, percentage in zip(sorted_labels, sorted_counts, sorted_percentages):
            print(f"類別 {label}: 佔比 {percentage:.2%}")
            cluster_label_df = cluster_label_df.append({
                'Cluster': cluster_idx, 
                'Class': label, 
                'Count': count,
                'Percentage': percentage
            }, ignore_index=True)

    # 將 Percentage 列的百分比格式化為字符串
    cluster_label_df['Percentage'] = cluster_label_df['Percentage'].map(lambda x: '{:.0%}'.format(x))

    return cluster_label_df  


def save_clustered_images(file_names, image_list, predictions, output_folder):
    """
    將分群後的影像保存到不同的資料夾中。

    Args:
    - file_names (list): 包含影像檔案名稱的列表。
    - image_list (list): 包含影像數據的列表。
    - predictions (list): 包含每個影像所屬的 cluster 的列表。
    - output_folder (str): 輸出資料夾的路徑。

    Returns:
    - None
    """
    # 如果輸出資料夾已存在，刪除它
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    # 創建輸出資料夾
    os.makedirs(output_folder)
    
    # 創建每個 cluster 的資料夾
    cluster_folders = {}
    for cluster_id in set(predictions):
        cluster_folder = os.path.join(output_folder, f'cluster_{cluster_id}')
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        cluster_folders[cluster_id] = cluster_folder

    # 將影像保存到對應的 cluster 資料夾中
    for (file_name, image, cluster_id) in zip(file_names, image_list, predictions):
        cluster_folder = cluster_folders[cluster_id]

        # 假設影像是 NumPy 陣列，且值範圍是 0-255
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # 將 NumPy 陣列轉換為 PIL 圖像
        image_pil = Image.fromarray(image)
        image_path = os.path.join(cluster_folder, f'{file_name}.png')
        image_pil.save(image_path)



#################################################################

combined_names, combined_list = read_npy_files(folder_path)
combined_GT_list = read_ground_truth(folder_path)
rice_names, rice_list, non_rice_names, non_rice_list = split_npy_files_by_ground_truth(combined_names, combined_list, combined_GT_list)

# print("Rice list length:", len(rice_list))
# print("Non-rice list length:", len(non_rice_list))


print("start combined clustering")

combined_outputs = do_Contrastive_learning(image_list=combined_list, saved_model_path=saved_model_path)
combined_predictions = do_Kmeans_clustering(image_list=combined_outputs,
                                        clusters_amount=clusters_amount)  # 分群結果 屬於哪個cluster

# 保存分群後的影像
save_clustered_images(combined_names, combined_list, combined_predictions, subfolder_path)
# 可視化 ground truth
do_tsne(combined_outputs, combined_GT_list, "t-SNE Visualization of Ground Truth")
# 可視化 K-means 聚類結果
do_tsne(combined_outputs, combined_predictions, "t-SNE Visualization of cluster result")

print("end combined clustering")

# 統計每個簇中的類別分佈
cluster_label_counts = calculate_cluster_label_distribution(combined_predictions, np.array(combined_GT_list), num_clusters=clusters_amount)

# 創建包含類別分佈的 DataFrame
cluster_label_df = create_distribution_dataframe(cluster_label_counts)

# 使用 pivot_table 函數將 DataFrame 重塑為矩陣形式
pivot_table = cluster_label_df.pivot_table(index='Class', columns='Cluster', values='Percentage', aggfunc='first')

# 將 DataFrame 保存為 CSV 文件
pivot_table.to_csv(os.path.join(subfolder_path, 'cluster_label_distribution.csv'), encoding="utf-8-sig")

print("CSV file 'cluster_label_distribution.csv' has been created.")


# print("start Rice Clustering")

# rice_outputs = do_Contrastive_learning(image_list=rice_list, saved_model_path=saved_model_path)
# rice_predictions = do_Kmeans_clustering(image_list=rice_outputs,
#                                         clusters_amount=clusters_amount)  # 分群結果 屬於哪個cluster

# # 保存分群後的影像
# save_clustered_images(rice_names, rice_list, rice_predictions, subfolder_path+"/rice_clusters")

# # 可視化 K-means 聚類結果
# do_tsne(rice_outputs, rice_predictions, "t-SNE Visualization of simclr + Kmeans")

# print("end Rice Clustering")

# print("start Non Rice Clustering")

# non_rice_outputs = do_Contrastive_learning(image_list=non_rice_list, saved_model_path=saved_model_path)
# non_rice_predictions = do_Kmeans_clustering(image_list=non_rice_outputs,
#                                         clusters_amount=clusters_amount)  # 分群結果 屬於哪個cluster

# # 保存分群後的影像
# save_clustered_images(non_rice_names, non_rice_list, non_rice_predictions, subfolder_path+"/non_rice_clusters")

# # 可視化 K-means 聚類結果
# do_tsne(non_rice_outputs, non_rice_predictions, "t-SNE Visualization of simclr + Kmeans")

# print("end Non Rice Clustering")

