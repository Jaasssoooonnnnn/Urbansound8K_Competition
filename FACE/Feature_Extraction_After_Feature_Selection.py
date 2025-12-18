import pandas as pd
import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os

# ================= 1. 配置绝对路径 (根据你提供的信息) =================
# CSV 文件路径
CSV_PATH = '/your_path/Kaggle_Data/metadata/kaggle_train.csv'

# 音频根目录 (即 fold1, fold2... 所在的目录)
AUDIO_ROOT = '/your_path/Kaggle_Data/audio'

# ================= 2. 读取数据 =================
# 强制将 fold 读取为字符串(str)，避免后续拼接路径时报错
df = pd.read_csv(CSV_PATH, dtype={'slice_file_name': str, 'fold': str, 'classID': int})

# 提取必要列
data_records = df[['slice_file_name', 'fold', 'classID']].values

# 初始化特征数组和标签数组
# 这里我们动态获取行数，避免 index out of bounds 错误。
n_samples = len(data_records)
print(f"Total samples to process: {n_samples}")

features = np.zeros(shape=(n_samples, 810), dtype=np.float32)
labels = np.zeros(shape=(n_samples,), dtype=np.int64)

# ================= 3. 定义特征提取函数 =================
def pack_features(extracted_feature):
    delta = gaussian_filter1d(extracted_feature, sigma=1, order=1, mode='nearest')
    delta_delta = gaussian_filter1d(extracted_feature, sigma=1, order=2, mode='nearest')
    
    mean_vector = np.concatenate(
        (np.mean(extracted_feature, axis=1), np.mean(delta, axis=1), np.mean(delta_delta, axis=1)))
    var_vector = np.concatenate(
        (np.var(extracted_feature, axis=1), np.var(delta, axis=1), np.var(delta_delta, axis=1)))
    
    feature_vector = np.concatenate((mean_vector, var_vector))
    return feature_vector

# ================= 4. 主循环 =================
print("Starting feature extraction...")

for index, (filename, fold, class_id) in enumerate(data_records):
    # 打印进度
    if index % 100 == 0:
        print(f"Processing {index}/{n_samples}")

    # 拼接绝对路径
    # 逻辑：AUDIO_ROOT + foldX + filename
    # 这里的 fold 已经是字符串 "1", "2" 等，所以 fold_dir 是 "fold1"
    fold_dir = f"fold{fold}" 
    file_path = os.path.join(AUDIO_ROOT, fold_dir, filename)
    
    try:
        # 加载音频 (指定 sr=22050 保证一致性)
        audio, sample_rate = librosa.load(file_path, sr=22050)

        # 提取特征
        mfc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128, n_fft=1024)
        cnt = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)

        # 存入数组
        features[index, 0:768] = pack_features(mfc)
        features[index, 768:810] = pack_features(cnt)
        
        # 存入标签 (你的原代码漏了这一步)
        labels[index] = class_id

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # 如果出错，通常填 0 或者记录下来后续删除，这里保持 0

# ================= 5. 保存结果 =================
print("Saving output files...")
np.save("train_features.npy", features)
np.save("train_labels.npy", labels)
print("All done!")