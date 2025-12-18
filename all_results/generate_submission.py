import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

# =========================
# 1. 配置：12个模型列表
# =========================
CHOSEN_MODELS = [
    "resnet34_mr_salience_fusion_film_attention/best_model_fold1.csv",
    "attention/best_model_fold2.csv",
    "resnet34_mr_salience_fusion_film_attention/best_model_fold4.csv",
    "wideresnet/best_model_fold5.csv",
    "res2net50/best_model_fold7.csv",
    "attention/best_model_fold8.csv",
    "resnet34_pcen_sam_8fold/best_model_fold1.csv",
    "resnet34_pcen_sam_8fold/best_model_fold6.csv",
    "resnet34_salience_fusion_film_attention/best_model_fold1.csv",
    "resnet_optuna/best_model_fold2.csv",
    "fusion_128/best_model_fold2.csv",
    "resnet_optuna/best_model_fold7.csv"
     "resnet_optuna/best_model_fold1.csv",
      "best_resnet/best_model_fold5.csv"
]

NUM_CLASSES = 10

# =========================
# 2. 路径设置
# =========================
def find_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        # 自动寻找项目根目录
        if (cur / "Kaggle_Data").exists() or (cur / "urbansound8k").exists():
            return cur
        cur = cur.parent
    return Path(".")

ROOT = find_root(Path(__file__).parent)
# 如果找不到文件，请手动修改这个 PREDICTION_ROOT 为存放 CSV 的最上级目录
PREDICTION_ROOT = ROOT 
ID_MAP_PATH = ROOT / "Kaggle_Data" / "metadata" / "kaggle_test.csv"

# =========================
# 3. 核心逻辑
# =========================
def find_file(filename, search_root):
    """递归查找文件，防止路径层级不对"""
    # 1. 尝试直接路径
    direct_path = search_root / filename
    if direct_path.exists():
        return direct_path
    
    # 2. 尝试递归搜索
    print(f"🔍 Searching for {filename}...")
    found = list(search_root.rglob(filename.split('/')[-1])) # 只搜文件名
    if found:
        # 如果有多个同名文件，尝试匹配父目录
        for f in found:
            if str(f).endswith(filename):
                return f
        return found[0] # 没匹配到路径，就返回第一个同名的
    
    raise FileNotFoundError(f"❌ Could not find file: {filename}")

def main():
    print(f"🚀 Generating submission for {len(CHOSEN_MODELS)} models...")
    
    # 1. 加载 ID 列表 (Submission 模板)
    if not ID_MAP_PATH.exists():
        print(f"Error: Metadata file not found at {ID_MAP_PATH}")
        return
        
    test_df = pd.read_csv(ID_MAP_PATH)
    ids_ref = test_df["ID"].values
    print(f"📋 Target Samples: {len(ids_ref)}")
    
    # 初始化概率矩阵 (N, 10)
    total_probs = np.zeros((len(ids_ref), NUM_CLASSES), dtype=np.float32)
    
    # 2. 逐个模型读取并累加
    loaded_count = 0
    prob_cols = [str(i) for i in range(NUM_CLASSES)]
    
    for model_name in CHOSEN_MODELS:
        try:
            full_path = find_file(model_name, PREDICTION_ROOT)
            print(f"   Reading: {model_name}")
            
            df = pd.read_csv(full_path)
            df["ID"] = df["ID"].astype(int)
            df = df.set_index("ID")
            
            # 对齐 ID
            probs = df.loc[ids_ref, prob_cols].values.astype(np.float32)
            
            # 归一化 (Softmax 概率和应为 1)
            row_sum = probs.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0 # 避免除以0
            probs = probs / row_sum
            
            # 累加 (Soft Voting)
            total_probs += probs
            loaded_count += 1
            
        except Exception as e:
            print(f"⚠️ Error processing {model_name}: {e}")
            
    if loaded_count != len(CHOSEN_MODELS):
        print(f"⚠️ Warning: Only loaded {loaded_count}/{len(CHOSEN_MODELS)} models!")
    
    # 3. 生成最终预测 (Argmax)
    print("🧮 Calculating final predictions...")
    final_preds = total_probs.argmax(axis=1)
    
    # 4. 保存 CSV
    submission = pd.DataFrame({
        "ID": ids_ref,
        "Target": final_preds
    })
    
    out_file = "submission.csv"
    submission.to_csv(out_file, index=False)
    
    print(f"\n✅ Submission saved to: {out_file}")
    print(submission.head())

if __name__ == "__main__":
    main()