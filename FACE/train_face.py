import os
# ================= 强制使用 Legacy Keras =================
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# =======================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU, Add, Conv1D, Flatten, BatchNormalization, LocallyConnected1D, GaussianNoise
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import QuantileTransformer
from scipy.ndimage import gaussian_filter1d
import librosa
import gc

# ================= 1. 全局配置 =================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 路径配置
BASE_PATH = os.getcwd()
DATA_ROOT = '/your_path/Kaggle_Data'
AUDIO_ROOT = os.path.join(DATA_ROOT, 'audio') # 确保音频在这里
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, 'metadata', 'kaggle_train.csv')
TEST_CSV_PATH = os.path.join(DATA_ROOT, 'metadata', 'kaggle_test.csv')

# 特征缓存路径
TRAIN_FEATS_PATH = os.path.join(BASE_PATH, "train_features.npy")
TRAIN_LABELS_PATH = os.path.join(BASE_PATH, "train_labels.npy")
TEST_FEATS_PATH = os.path.join(BASE_PATH, "test_features.npy") # 测试集特征缓存

# 超参数
BATCH_SIZE = 32
EPOCHS = 80 
LEARNING_RATE = 0.01

# 数据划分
TRAIN_FOLDS = ['1', '2', '3', '4', '5', '6', '7']
VAL_FOLDS = ['8']

# ================= 2. 特征提取工具 (用于测试集) =================
def pack_features(extracted_features):
    """将提取的各个特征拼接成向量"""
    mean_vector = np.concatenate([np.mean(f, axis=1) for f in extracted_features])
    var_vector = np.concatenate([np.var(f, axis=1) for f in extracted_features])
    return np.concatenate((mean_vector, var_vector))

def extract_features_from_file(file_path):
    """处理单个音频文件"""
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050)
        
        # 特征提取逻辑 (必须与训练集完全一致)
        mfc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128, n_fft=1024)
        cnt = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_fft=1024)
        chr = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_fft=1024)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=1024)
        ton = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
        
        # Delta 特征
        delta = gaussian_filter1d(mfc, sigma=1, axis=1, order=1)
        delta_delta = gaussian_filter1d(mfc, sigma=1, axis=1, order=2)
        
        return pack_features([mfc, cnt, chr, mel, ton, delta, delta_delta])
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return np.zeros(810)

def prepare_test_features():
    """检查缓存，如果没有则提取测试集特征"""
    if os.path.exists(TEST_FEATS_PATH):
        print("Loading cached test features...")
        return np.load(TEST_FEATS_PATH)
    
    print("Extracting Test Features (this may take a while)...")
    df_test = pd.read_csv(TEST_CSV_PATH)
    feats = []
    
    total = len(df_test)
    for idx, row in df_test.iterrows():
        if idx % 100 == 0: print(f"Processing {idx}/{total}...")
        
        # 寻找音频文件
        fname = row['slice_file_name']
        # 测试集音频可能在 audio/test/ 或者 audio/ 下，尝试寻找
        path1 = os.path.join(AUDIO_ROOT, 'test', fname)
        path2 = os.path.join(AUDIO_ROOT, fname)
        
        if os.path.exists(path1):
            feats.append(extract_features_from_file(path1))
        elif os.path.exists(path2):
            feats.append(extract_features_from_file(path2))
        else:
            print(f"Warning: File {fname} not found! Filling with zeros.")
            feats.append(np.zeros(810))
            
    feats = np.array(feats, dtype=np.float32)
    np.save(TEST_FEATS_PATH, feats)
    print("Test features saved.")
    return feats

# ================= 3. 数据加载与分割 =================
def load_data_and_scaler():
    print("Loading Train features...")
    all_features = np.load(TRAIN_FEATS_PATH)
    all_labels = np.load(TRAIN_LABELS_PATH)
    
    df = pd.read_csv(TRAIN_CSV_PATH)
    folds = df['fold'].astype(str).values
    
    train_mask = np.isin(folds, TRAIN_FOLDS)
    val_mask = np.isin(folds, VAL_FOLDS)
    
    X_train = all_features[train_mask]
    y_train = all_labels[train_mask]
    X_val = all_features[val_mask]
    y_val = all_labels[val_mask]
    
    print("Fitting QuantileTransformer on Train Data...")
    scaler = QuantileTransformer(n_quantiles=5000, output_distribution='uniform')
    
    # 训练 Scaler 并转换训练集
    X_train = scaler.fit_transform(X_train)
    # 转换验证集
    X_val = scaler.transform(X_val)
    
    # 数据整形
    X_train = X_train * 2 - 1
    X_val = X_val * 2 - 1
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    
    # One-Hot 标签
    y_train_hot = to_categorical(y_train, num_classes=10)
    y_val_hot = to_categorical(y_val, num_classes=10)
    
    return X_train, y_train_hot, X_val, y_val_hot, scaler

# ================= 4. Mixup 生成器 =================
def mixup_generator(X, y, batch_size=32, alpha=0.3):
    while True:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i+batch_size]
            if len(batch_indices) < batch_size: continue
            
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            lam = np.random.beta(alpha, alpha, batch_size)
            lam_X = lam.reshape(-1, 1, 1)
            lam_y = lam.reshape(-1, 1)
            
            perm_indices = np.random.permutation(batch_size)
            X_mix = lam_X * X_batch + (1 - lam_X) * X_batch[perm_indices]
            y_mix = lam_y * y_batch + (1 - lam_y) * y_batch[perm_indices]
            
            yield X_mix, y_mix

# ================= 5. 模型定义 =================
def build_face_model(input_shape, num_classes=10):
    f_inp = Input(shape=input_shape)
    x = GaussianNoise(0.1)(f_inp) # 噪声增强
    reg = l2(0.0005) # L2 正则
    
    streams = []
    for k in [3, 7, 11, 17, 23]:
        x_s = Conv1D(64, k, padding='same', kernel_regularizer=reg)(x)
        x_s = BatchNormalization()(x_s); x_s = LeakyReLU(alpha=0.05)(x_s)
        x_s = Conv1D(64, k, padding='same', kernel_regularizer=reg)(x_s)
        x_s = BatchNormalization()(x_s); x_s = LeakyReLU(alpha=0.05)(x_s)
        x_s = LocallyConnected1D(16, 3, kernel_regularizer=reg)(x_s)
        x_s = BatchNormalization()(x_s); x_s = LeakyReLU(alpha=0.05)(x_s)
        x_s = Conv1D(16, 3, padding='same', kernel_regularizer=reg)(x_s)
        x_s = BatchNormalization()(x_s); x_s = LeakyReLU(alpha=0.05)(x_s)
        streams.append(x_s)
    
    x = Add()(streams)
    x = LocallyConnected1D(32, 3, kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = LeakyReLU(alpha=0.05)(x)
    x = Conv1D(32, 3, padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = LeakyReLU(alpha=0.05)(x)
    
    x = Flatten()(x)
    x = Dense(200, kernel_regularizer=reg)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Dropout(0.6)(x) # 高 Dropout
    x_out = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=f_inp, outputs=x_out)

# ================= 6. 主程序 =================
if __name__ == "__main__":
    # --- A. 准备数据 (不训练也要加载数据来做归一化) ---
    # 注意：这里必须要加载Scaler，否则测试集没办法做正确的归一化
    X_train, y_train, X_val, y_val, scaler = load_data_and_scaler()
    
    # 构建模型结构
    model = build_face_model(input_shape=(810, 1), num_classes=10)
    
    # 【重要】即使不训练，如果后面要运行 model.evaluate，也必须 compile
    # 如果你只想生成CSV不想看验证集分数，可以把 evaluate 那两行删掉，那样就不需要 compile 了
    model.compile(optimizer='sgd', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # --- 跳过训练阶段 (代码已注释，无需修改) ---
    # history = model.fit(...) 
    
    # --- B. 验证阶段 (可选) ---
    print(f"\n{'='*20} Validation Report (Fold 8) {'='*20}")
    try:
        # 加载你已经训练好的权重
        model.load_weights('best_face_model_optimized.h5')
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Accuracy: {acc:.4f}")
    except OSError:
        print("错误：找不到 'best_face_model_optimized.h5' 文件。请确认你已经训练过并保存了模型。")
        exit()
    
    # 释放内存
    del X_train, y_train, X_val, y_val
    gc.collect()
    
    # --- C. 推理与提交阶段 ---
    print(f"\n{'='*20} Generating Submission {'='*20}")
    
    # 1. 准备测试特征
    X_test = prepare_test_features()
    
    # 2. 归一化 (使用训练好的 scaler)
    X_test = scaler.transform(X_test)
    X_test = X_test * 2 - 1
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # 3. 预测
    print("Predicting...")
    preds = model.predict(X_test, verbose=1)
    pred_classes = np.argmax(preds, axis=1)
    
    # 4. 【关键修改】写入符合 ID,TARGET 格式的 CSV
    print("Saving submission...")
    
    submission = pd.DataFrame({
        'ID': range(len(pred_classes)),  # 生成 0, 1, 2, ... 索引
        'TARGET': pred_classes           # 预测类别
    })
    
    save_name = 'submission_optimized.csv'
    submission.to_csv(save_name, index=False)
    print(f"Done! Submission saved to {save_name}")
    print("Format check (First 5 rows):")
    print(submission.head())