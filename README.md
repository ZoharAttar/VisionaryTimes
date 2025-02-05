# VisionaryTimes - Using LLMs for Time-Series Analysis (Our Final Reasearch Project)

## Our project goal: 
**Enhancing the capabilities of TEMPO (Prompt-based Generative Pre-trained Transformer for Time Series Forecasting) by:**

1. **Integrating multimodal inputs (Vision+Text) in TEMPO** - We plan to achieve this by integrating data through tools like CLIP (Contrastive Language–Image Pre-training), to enhance the model’s performance in handling complex scenarios.
2. **Adjust TEMPO for classification task** - we aim to improve its accuracy and versatility while exploring its potential for handling additional tasks such as classification.
3. **Use Pre-Training techniques to enhance TEMPO performance** - we will focus on pre-training techniques, including masking strategies, to improve the model’s ability to learn complex temporal patterns and boost its generalization potential.

The official code for [["TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting (ICLR 2024)"]](https://arxiv.org/pdf/2310.04948).

TEMPO is one of the very first open source **Time Series Foundation Models** for forecasting task v1.0 version.

<div align="center"><img src=https://raw.githubusercontent.com/DC-research/TEMPO/main/tempo/pics/TEMPO.png width=80% /></div>


## Add vision to TEMPO Architecture 

![image](https://github.com/user-attachments/assets/7406234a-befe-4696-a70b-f410aae2ad55)

**Inserted CLIP embedding in addition to prompt**
- Image generation for each component (trend, seasonality, residual)
- Encoding with CLIP’s image encoder
- Concatenate to prompt and TS embedding

**Train the same data as it done originaly in TEMPO, to check visual input’s effect**

**We invite you to read in detail about the adjustments we have made so far, as described in the following attached document:**

"VisionaryTimes - Summary of the main progress in the BGU hackathon"

## What’s Next:
1. Try different implementations for Visual Inputs usage:
	- Feed global input to CLIP, get patches vectors and then concatenate
	- Feed patches input to CLIP, get vectors and then concatenate

2. Adjust TEMPO for classification task

3. Use Pre-Training techniques to enhance the model performance


### Get Data

   Download the data from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), and place the downloaded data in the folder`./dataset`. You can also download the STL results from [[Google Drive]](https://drive.google.com/file/d/1gWliIGDDSi2itUAvYaRgACru18j753Kw/view?usp=sharing), and place the downloaded data in the folder`./stl`.

### Run TEMPO

### Pre-Training Stage
```
bash [ecl, etth1, etth2, ettm1, ettm2, traffic, weather].sh
```

### Test/ Inference Stage

After training, we can test TEMPO model under the zero-shot setting:

```
bash [ecl, etth1, etth2, ettm1, ettm2, traffic, weather]_test.sh
```

### TEMPO Results

<div align="center"><img src=https://raw.githubusercontent.com/DC-research/TEMPO/main/tempo/pics/results.jpg width=90% /></div>






