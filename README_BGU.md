## Intro
This code is based on the TEMPO repositiry: https://github.com/databrickslabs/tempo

## insall packeges if nedeed
```
pip install -r requirements.txt
```

## Build the environment

```
conda create -n tempo python=3.8
```
```
conda activate tempo
```
```
pip install timeagi
```

## Add data - one off
1. create a folder called ```dataset```
2. download the datasets from temp google drive: [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) 
3. put the datasets inside the folder

## Run the Model (Train the whole model)
```
python train_TEMPO.py
```
## Run the Model on specific satasets:
```
bash (specific dataset file directory)
for example:
bash C:\......\final project\VisionaryTimes-1\scripts\etth2.sh
```

