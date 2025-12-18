# /your_path/TSCNN/config.py

import os

class Config:
    # ================= 路径设置 =================
    # 原始数据集路径
    DATASET_ROOT = "/your_path/urbansound8k"
    METADATA_PATH = os.path.join(DATASET_ROOT, "UrbanSound8K.csv")
    AUDIO_DIR = DATASET_ROOT # 假设音频在 fold1-fold10 子文件夹中

    # 处理后数据的保存路径
    PROCESSED_DATA_DIR = "/your_path/TSCNN/processed_data"
    
    # ================= 音频处理参数 (Paper Section 4) =================
    SR = 22050           #
    N_FFT = 512          # ~23ms frame size
    HOP_LENGTH = 256     # 50% overlap
    N_MELS = 60          # Log-Mel channels
    N_MFCC = 20          # Base MFCCs (will become 60 with deltas)
    
    # ================= 维度设置 =================
    # 论文要求输入维度为 41 x 85 (Time x Features)
    # 我们这里定义固定帧数
    FIXED_FRAMES = 41    #
    
    # CST 特征维度
    N_CHROMA = 12        #
    N_CONTRAST = 7       #
    N_TONNETZ = 6        #
    
    # ================= 训练/测试划分 =================
    # 严格按照您的要求：Fold 1-8 训练，Fold 9-10 测试
    TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
    TEST_FOLDS = [9, 10]
    
    # ================= 训练超参数 (Paper Section 3.2) =================
    BATCH_SIZE = 32      #
    LEARNING_RATE = 0.001 #
    MOMENTUM = 0.9       #
    EPOCHS = 50          # 论文未明确Epochs，根据经验设定，通常50-100足够
    DROPOUT = 0.5        #
    
    # 创建目录
    @staticmethod
    def setup_dirs():
        os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)

# 初始化目录
if __name__ == "__main__":
    Config.setup_dirs()
    print(f"Directories checked/created at {Config.PROCESSED_DATA_DIR}")