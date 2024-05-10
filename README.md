# Toward Availability Attacks in 3D Point Clouds
This is the official repo of the ICML 2024 paper "Toward Availability Attacks in 3D Point Clouds".

## Requirements

Create environments for FC-EM:
```shell
conda create -n fc-em python=3.9
 
pip install -r requirements.txt

conda activate fc-em
```

## Download Dataset

### ModelNet40
https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
### ScanObjectNN
https://hkust-vgd.ust.hk/scanobjectnn/
### IntrA
https://drive.google.com/drive/folders/1yjLdofRRqyklgwFOC0K4r7ee1LPKstPh/IntrA.zip
### BFM2017
https://faces.dmi.unibas.ch/bfm/bfm2017.html

After download and uncompress dataset, move them to ```/data``` folder.

## Running Experiments

### Run ModelNet40
```shell
python main.py --exp-config configs/modelnet40/fc_errmin/fc_em_gen.yaml --gpu-id {gpu} --chamf_coeff {beta} 

python main.py --exp-config configs/modelnet40/fc_errmin/fc_em_eval.yaml --gpu-id {gpu} --chamf_coeff {beta} 
```


### Run ScanObjectNN
```shell
python main.py --exp-config configs/extra/scan/scan_gen.yaml --gpu-id {gpu} --chamf_coeff {beta} 

python main.py --exp-config configs/extra/scan/scan_eval.yaml --gpu-id {gpu} --chamf_coeff {beta} 
```


### Run IntrA
```shell
python main_medi.py --exp-config configs/extra/intra/intra_gen.yaml --gpu-id {gpu} --chamf_coeff {beta} 

python main_medi.py --exp-config configs/extra/intra/intra_eval.yaml --gpu-id {gpu} --chamf_coeff {beta} 
```


### Run Generated Face 
```shell
python main_medi.py --exp-config configs/extra/face/bsm2017_gen.yaml --gpu-id {gpu} --chamf_coeff {beta} 

python main_medi.py --exp-config configs/extra/face/bsm2017_eval.yaml --gpu-id {gpu} --chamf_coeff {beta} 
```


### Run Defense
```shel
cd configs/defense

sh run_fc_em_defense.sh
```