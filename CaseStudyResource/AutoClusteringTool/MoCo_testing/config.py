import os

folder_path = r'.\data\new_data'
saved_model_path = r'C:\Users\Jackson\Desktop\MoCo_training\result\v1\240926_Thu_13_55_21\model\moco_model_64_3000_0.03'

#　clusters_amount = 4
clusters_amount = {{ clusters_amount }}

subfolder_path = os.path.join(folder_path, "cluster"+str(clusters_amount))
