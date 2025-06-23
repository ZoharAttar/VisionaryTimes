# VisionaryTimes - Foundational Multi-Modal Time Series Model

## Our project goal: 
**Adapt an LLM-based foundational model, currently used for time series forecasting (TEMPO) to:**

1. **Multi-Modal** - Integrate visual representations of time series data using visual encoders
2. **Multi-Task** - Extend and evaluate foundational models, mainly focused on forecasting, to uni-modal and multi-modal classification


The official code for [["TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting (ICLR 2024)"]](https://arxiv.org/pdf/2310.04948).

TEMPO is one of the very first open source **Time Series Foundation Models** for forecasting task v1.0 version.

<div align="center"><img src=https://raw.githubusercontent.com/DC-research/TEMPO/main/tempo/pics/TEMPO.png width=80% /></div>


## Adding Visual Embedding Process
 
![image](https://github.com/user-attachments/assets/8380e8c0-b85d-4b03-b37a-94074a2e971b)

**Inserted Visual embedding in addition to prompt**
- Image generation for each component (trend, seasonality, residual)
- Encoding with image encoder
- Concatenate to prompt and TS embedding

## Evaluate Visual Embedding Quality
Exhibit different visual-encoders (CLIP, ViT, DeiT-Tiny, ResNet, DINOv2, De-Plot, and SigLIP) to evaluate Visual Embedding Quality by comparing their visual embeddings using multiple techniques:
- Cosine Similarity
- t-SNE Visualization
- Silhouette Score Clustering

## Results

**Train the same data as it done originaly in TEMPO, to check visual input’s effect**

### **Vision Results on Forecasting**

Comparison of MSE of vision encoders across in-domain and Zero-Shot long-term forecasting

![image](https://github.com/user-attachments/assets/67a6b810-02d2-481e-93ed-f0adbb402b81)

### **Classification Results**

Comparison of Accuracy scores across classification datasets with TimesNet (SOTA TS classification model without multi-modal capabilities)
![image](https://github.com/user-attachments/assets/4b4f5ad0-0b04-41f5-b897-d3d82d606103)

![image](https://github.com/user-attachments/assets/e1b805c8-ea77-4484-88e0-acda6e8cb014)

## Research Conclusions
1) Adding visual representations improves results in forecasting task
2) Adapting TEMPO for classification tasks achieves performance close to TimesNet, and in some cases, slightly surpasses it
3) Adding visual representations improves results in classification task in some cases
4) STL doesn’t improve classification results in current data architecture
5) The image embedder affects the model’s results -> CLIP < DeiT-Tiny < ViT  


**We invite you to read in detail about the adjustments we have made so far, as described in the following attached document:**

"VisionaryTimes - Summary of the main progress in the BGU hackathon"


### Get Data

   Download the data from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), and place the downloaded data in the folder`./dataset`. You can also download the STL results from [[Google Drive]](https://drive.google.com/file/d/1gWliIGDDSi2itUAvYaRgACru18j753Kw/view?usp=sharing), and place the downloaded data in the folder`./stl`.








