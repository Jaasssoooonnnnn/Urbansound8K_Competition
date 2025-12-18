This document summarizes the technical evolution and core configurations of the 8 key models used to achieve **1st Place** in the audio classification competition.

---

### 1. `resnet34_pcen_sam_8fold.py` (The Robust Baseline)

* **Key Innovations:**
* **Trainable PCEN:** Replaced traditional Log-Mel Spectrograms with **Per-Channel Energy Normalization (PCEN)** to handle gain variations and background noise. Optimized using `@torch.jit.script` to bypass Python loop bottlenecks.
* **SAM (Sharpness-Aware Minimization):** Implemented a two-step optimization to find parameters in "flat" loss regions, significantly improving **Generalization** on the test set.
* **Data Augmentation:** Combined **Mixup** (linear interpolation of samples/labels) and **SpecAugment** (Frequency/Time masking) to prevent overfitting.
* **Architecture Tweaks:** Modified ResNet-34 input channel to 1, added `BatchNorm2d` before the backbone, and adjusted the FC layer for 10-class classification.
* **8-Fold Cross-Validation:** Ensured maximum data utility and provided a stable foundation for later ensembling.



### 2. `resnet34_pcen_sam_8f_200e.py` (Hyperparameter Optimization)

* **Key Improvements:**
* **Optuna Integration:** Conducted over 300 trials of hyperparameter searching to find the optimal configuration.
* **Critical Insight:** While CV accuracy improved by ~8%, the test set only rose by 0.3%. This revealed that 8-fold CV can be misleading on **imbalanced datasets**, prompting a shift toward more sophisticated architectural features.



### 3. `resnet34_attention.py` (Attention Mechanisms)

* **Key Innovations:**
* **CBAM (Convolutional Block Attention Module):** Integrated **Channel Attention** (identifying important frequencies) and **Spatial Attention** (locating key time-frequency regions) after each ResNet layer.
* **ASP (Attentive Statistics Pooling):** Replaced Global Average Pooling (GAP) with ASP. It uses an attention network to calculate the **weighted mean** and **weighted standard deviation** across time frames, capturing the non-stationary nature of audio signals.



### 4. `resnet34_pcen_salience_fusion_attention.py` (Metadata Fusion)

* **Performance:** 87.7 (Single), 89.0 (Ensemble).
* **Motivation:** Using **Salience** metadata (Foreground vs. Background) to distinguish between overlapping sounds (e.g., a dog bark over air conditioner noise).
* **Key Innovations:**
* **Salience Injection:** Utilized `nn.Embedding(2, 64)` to map salience labels into high-dimensional vectors.
* **Embedding-level Mixup:** Specifically designed a function to linearly mix salience embedding vectors during training, ensuring mathematical consistency for the SAM optimizer.
* **Multi-source Concatenated Head:** Concatenated the 1024D ASP features with the 64D Salience vector for a final 1088D input to the classifier.



### 5. `resnet34_salience_fusion_film_attention.py` (Deep Interaction)

* **Key Innovations:**
* **FiLM (Feature-wise Linear Modulation):** Moved beyond simple concatenation. Used Salience embeddings to generate scaling (\gamma) and shifting (\beta) parameters to modulate the ResNet feature maps channel-wise.
* **Waveform Augmentation:** Added **Gaussian Noise** and **Gain** adjustment directly to the raw waveform before conversion to spectrograms.
* **No-MaxPool ResNet:** Replaced the initial MaxPool with `nn.Identity()`. This prevents excessive compression of high-resolution time-frequency details, which is crucial for short audio clips.



### 6. `resnet34_mr_salience_fusion_film_attention.py` (Multi-Resolution CRNN)

* **Key Innovations:**
* **Multi-Resolution PCEN:** Implemented dual paths using different FFT sizes—**2048 (High Frequency Resolution)** and **1024 (High Temporal Resolution)**. Features are concatenated in the channel dimension.
* **BiGRU Layer:** Transformed the architecture into a **CRNN**. Added a Bidirectional GRU after the ResNet backbone to model **Long-term Temporal Dependencies**, essential for periodic or evolving sounds.



### 7 & 8. `wideresnet.py` and `res2net50.py` (Advanced Backbones)

* **Common Improvement: Deep Audio Stem:**
* Replaced the standard 7 \times 7 convolution with a **Sequential 3-layer 3 \times 3 convolution** stack. This enhances the extraction of fine-grained spectral textures while reducing parameters.


* **WideResNet-50-2:**
* Increased model width (2048 output channels) to capture highly complex acoustic patterns while maintaining gradient stability.


* **Res2Net-50:**
* **Multi-scale Features:** Utilized hierarchical residual-like connections within a single block. This creates **Granular Receptive Fields**, allowing the model to process both wide-band shocks and narrow-band tonal components simultaneously.



---

### Final Result

By selecting the **top 12 performing variations** from these 8 architectures and applying an **Ensemble (Average)** strategy, the final submission achieved a score of **91.3**.