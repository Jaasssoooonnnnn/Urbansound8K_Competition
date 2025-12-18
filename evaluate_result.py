import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ================= 配置路径 =================
# 1. 你的预测结果 (ID, TARGET)
SUBMISSION_PATH = "/your_path/all_results/final_submission.csv"

# 2. 竞赛提供的测试集列表 (ID -> 文件名 的映射表)
ID_MAP_PATH = "Kaggle_Data/metadata/kaggle_test.csv"

# 3. 官方完整标签 (文件名 -> 真实类别 的映射表)
GROUND_TRUTH_PATH = "urbansound8k/UrbanSound8K.csv"

# ================= 主程序 =================
def evaluate():
    print(f"Loading submission: {SUBMISSION_PATH}")
    print(f"Loading ID map:     {ID_MAP_PATH}")
    print(f"Loading GroundTruth:{GROUND_TRUTH_PATH}")

    # 1. 读取所有文件
    try:
        sub_df = pd.read_csv(SUBMISSION_PATH)    # cols: ID, TARGET
        map_df = pd.read_csv(ID_MAP_PATH)        # cols: ID, slice_file_name, ...
        gt_df = pd.read_csv(GROUND_TRUTH_PATH)   # cols: slice_file_name, classID, ...
    except FileNotFoundError as e:
        print(f"\n❌ Error: 找不到文件 -> {e}")
        print("请检查 `ID_MAP_PATH` 等路径是否正确")
        return

    # 2. 第一步合并：把预测结果 (ID) 和 文件名 关联起来
    try:
        # 确保 ID 都是 int 类型
        sub_df['ID'] = sub_df['ID'].astype(int)
        map_df['ID'] = map_df['ID'].astype(int)
        
        result_df = pd.merge(sub_df, map_df[['ID', 'slice_file_name']], on='ID', how='left')
    except KeyError:
        print("❌ Error: csv 中找不到 'ID' 或 'slice_file_name' 列")
        return

    # 3. 准备真实标签字典 {文件名: 真实类别}
    # 只需要 Fold 9 和 10 (假设 Kaggle Test 对应的是原始数据的 fold 9 和 10)
    gt_df = gt_df[gt_df['fold'].isin([9, 10])]
    gt_dict = dict(zip(gt_df['slice_file_name'], gt_df['classID']))

    # 4. 第二步匹配：通过文件名获取真实标签
    y_true = []
    y_pred = []
    
    missing_count = 0
    
    for _, row in result_df.iterrows():
        fname = row['slice_file_name']
        pred = row['TARGET']
        
        if fname in gt_dict:
            y_true.append(gt_dict[fname])
            y_pred.append(pred)
        else:
            missing_count += 1

    print(f"\nSuccessfully matched {len(y_true)} samples.")
    if missing_count > 0:
        print(f"⚠️ Warning: {missing_count} files in submission were not found in Official Fold 9/10.")

    if len(y_true) == 0:
        print("❌ Error: 没有成功匹配任何数据。")
        return

    # 5. 计算指标
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    final_score = 0.8 * acc + 0.2 * macro_f1

    # 获取类别名称 (确保顺序正确)
    # UrbanSound8K 的 classID 通常是 0-9，这里做一个简单的排序提取
    class_map = gt_df[['classID', 'class']].drop_duplicates().sort_values('classID')
    class_names = class_map['class'].tolist()

    # 6. 输出报告
    print("\n" + "="*40)
    print("       🎉 FINAL EVALUATION 🎉       ")
    print("="*40)
    print(f"✅ Accuracy  : {acc:.5f}  (80%)")
    print(f"✅ Macro F1  : {macro_f1:.5f}  (20%)")
    print("-" * 40)
    print(f"🏆 SCORE     : {final_score:.5f}")
    print("="*40)

    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # ================= 新增：绘制混淆矩阵 =================
    print("Generating Confusion Matrix Plot...")
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 绘图
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(f'Confusion Matrix\nAcc: {acc:.4f} | F1: {macro_f1:.4f}', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图片
    save_path = "confusion_matrix.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Confusion Matrix saved to: {save_path}")

if __name__ == "__main__":
    evaluate()