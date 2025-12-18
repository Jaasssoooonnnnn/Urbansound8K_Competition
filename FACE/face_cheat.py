import os
# 强制 Legacy 模式
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU, Add, Conv1D, Flatten, BatchNormalization, LocallyConnected1D
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split  # <--- 关键工具
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score

# ================= 配置 =================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

BASE_PATH = os.getcwd()
FEATURES_PATH = os.path.join(BASE_PATH, "train_features.npy")
LABELS_PATH = os.path.join(BASE_PATH, "train_labels.npy")

# ================= 复刻作者的“作弊”划分 =================
def load_and_split_like_author():
    print("Loading dataset...")
    features = np.load(FEATURES_PATH)
    labels = np.load(LABELS_PATH)
    
    # 1. 归一化 (作者是在 Split 之前做的全量归一化，这也算一点泄露，我们保持一致)
    print("Applying QuantileTransformer (Global)...")
    n_samples = len(features)
    n_qt = min(n_samples, 5000)
    
    qt_mfc = QuantileTransformer(n_quantiles=n_qt, output_distribution='uniform')
    features[:, 0:768] = qt_mfc.fit_transform(features[:, 0:768])
    
    qt_cnt = QuantileTransformer(n_quantiles=n_qt, output_distribution='uniform')
    features[:, 768:810] = qt_cnt.fit_transform(features[:, 768:810])
    
    features = features * 2 - 1
    
    # 2. 随机划分 (Random Split) - 这就是 98% 的秘密
    print("Splitting data RANDOMLY (Ignoring Folds)...")
    # random_state=75 是作者 notebook 里的参数
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=75)
    
    # Reshape
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, y_train, X_test, y_test

# ================= 模型定义 (保持不变) =================
def ConvBlock(inputs, filters, kernel_size):
    x = Conv1D(filters, kernel_size, padding='same')(inputs)
    x = BatchNormalization()(x); x = LeakyReLU(alpha=0.05)(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x); x = LeakyReLU(alpha=0.05)(x)
    return x

def LocalBlock(inputs, filters, kernel_size):
    x = LocallyConnected1D(filters, kernel_size)(inputs)
    x = BatchNormalization()(x); x = LeakyReLU(alpha=0.05)(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x); x = LeakyReLU(alpha=0.05)(x)
    return x

def DenseBlock(inputs, neuron_count):
    x = Dense(neuron_count)(inputs); x = LeakyReLU(alpha=0.05)(x); x = Dropout(0.5)(x)
    return x

def build_face_model(input_shape, num_classes=10):
    f_inp = Input(shape=input_shape)
    streams = []
    for k in [3, 7, 11, 17, 23]:
        x = ConvBlock(f_inp, 64, k); x = LocalBlock(x, 16, 3); streams.append(x)
    x = Add()(streams)
    x = LocalBlock(x, 32, 3); x = Flatten()(x)
    x = DenseBlock(x, 200)
    x_out = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=f_inp, outputs=x_out)

# ================= 训练 =================
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_and_split_like_author()
    
    model = build_face_model(input_shape=(810, 1), num_classes=10)
    
    # 作者参数: SGD, lr=0.01, 25 Epochs (我们跑个 40 epoch 看看)
    model.compile(optimizer=SGD(learning_rate=0.01), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    print("Starting training (Author's Setup)...")
    history = model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    probs = model.predict(X_test)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, preds)
    print(f"\n>>> LEAKAGE ACCURACY: {acc:.4f} <<<")