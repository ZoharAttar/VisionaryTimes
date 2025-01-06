## Intro
This code is based on the TEMPO repositiry: https://github.com/databrickslabs/tempo

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

## Run the Model
```
python train_TEMPO.py
```
## TODO:
1. understand each argument in train_TEMPO.py and the effects
2. create a bash file per dataset (example: BGU_bash/ETTh2.sh)
3. learn the TEMPO paper
