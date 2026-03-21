# A Unified Sketch Biometric Identification Framework

## Install

- Build environment

```bash
conda env create -f env.yaml
```

- Download checkpoints



- Download datasets

  In our experiment we use MaSk1k, CUFSF and IIIT-D Viewed Sketch datasets for evaluation. Because we do not own the datasets,  you need to download them yourself.

  After download you need to put them in the correct path as following.

  ```
  ├─Data_sets
  │  ├─CUFSF
  │  │  ├─photo
  │  │  │   ├─query
  │  │  │   └─train
  │  │  └─sketch
  │  │      ├─query
  │  │      └─train
  │  ├─IIIT-D_sketch
  │  │  ├─Photo
  │  │  │   ├─query
  │  │  │   └─train
  │  │  └─Sketch
  │  │      ├─query
  │  │      └─train
  │  └─MS1k
  │      ├─photo
  │      │  ├─query
  │      │  └─train
  │      └─sketch
  │          ├─query
  │          └─train
  ```

  

## Usage

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py  TEST.WEIGHT 'trained_checkpoints.pth'
```
