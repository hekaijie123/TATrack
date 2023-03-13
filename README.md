# TATrack

# Target-Aware Tracking with Long-term Context Attention has been accepted by AAAI23.

Law Result and Weights: https://drive.google.com/drive/folders/1PqiciVkwmtD9VCRkHVhZLsLA6cuz5oF1?usp=share_link

## Setup

* Create a new conda environment and activate it.
```Shell
conda create -n TATrack python=3.9 -y
conda activate TATrack
```

* Install `pytorch` and `torchvision`.
```Shell
conda install pytorch torchvision cudatoolkit -c pytorch

```

* Install other required packages.
```Shell
pip install -r requirements.txt
```

## Test
* Prepare the datasets: OTB2015, VOT2018, UAV123, GOT-10k, TrackingNet, LaSOT, COCO*, and something else you want to test. Set the paths as the following: 
```Shell
├── TATrack
|   ├── ...
|   ├── ...
|   ├── datasets
|   |   ├── COCO -> /opt/data/COCO
|   |   ├── GOT-10k -> /opt/data/GOT-10k
|   |   ├── LaSOT -> /opt/data/LaSOT/LaSOTBenchmark
|   |   ├── OTB
|   |   |   └── OTB2015 -> /opt/data/OTB2015
|   |   ├── TrackingNet -> /opt/data/TrackingNet
|   |   ├── UAV123 -> /opt/data/UAV123/UAV123
|   |   ├── VOT
|   |   |   ├── vot2018
|   |   |   |   ├── VOT2018 -> /opt/data/VOT2018
|   |   |   |   └── VOT2018.json
```
* Notes

> i. Star notation(*): just for training. You can ignore these datasets if you just want to test the tracker.
> 
> ii. In this case, we create soft links for every dataset. The real storage location of all datasets is `/opt/data/`. You can change them according to your situation.
> 


<!-- * Download the models we trained. -->
    

<!-- 
* Use the path of the trained model to set the `pretrain_model_path` item in the configuration file correctly, then run the shell command.
 -->

* Note that all paths we used here are relative, not absolute. See any configuration file in the `experiments` directory for examples and details.

### General command format
```Shell
python main/test.py --config testing_dataset_config_file_path
```

Take GOT-10k as an example:
```Shell
python main/test.py --config experiments/tatrack/test/base/got.yaml
```

## Training
* Prepare the datasets as described in the last subsection.
* Run the shell command.

### training based on the GOT-10k benchmark
```Shell
python main/train.py --config experiments/tatrack/train/base-got.yaml
```

### training with full data
```Shell
python main/train.py --config experiments/tatrack/train/base.yaml
```


# BibTeX
@article{he2023target,
  title={Target-Aware Tracking with Long-term Context Attention},
  author={He, Kaijie and Zhang, Canlong and Xie, Sheng and Li, Zhixin and Wang, Zhiwen},
  journal={arXiv preprint arXiv:2302.13840},
  year={2023}
}