# DDDAN

This repository is the official implementation of [DDDAN: Deep discriminative domain adaptation network considering sampling frequency for cross-domain mechanical fault diagnosis](https://www.sciencedirect.com/science/article/pii/S0957417425009182) (ESWA 2025).


## Requirements
- Python 3.9
- Numpy 1.16.2
- Pandas 0.24.2
- tqdm 4.31.1
- sklearn 0.21.3
- Scipy 1.2.1
- pytorch >= 1.2
- torchvision >= 0.40


## Datasets
- **[PHM 2009](https://www.phmsociety.org/competition/PHM/09/apparatus)**


## Usage
- use the `train.py` to train
- for example, use the following commands to test JDA_W for PHM with the transfer_task 0-->3
- `python train.py --data_name PHM --data_dir D:/Data/PHM --transfer_task [0],[3] --last_batch True --distance_metric True --distance_loss JDA_W`


## References
Part of the code refers to the following open source code:
- [SWK.py](https://github.com/liguge/WIDAN) from the paper "[Interpretable Physics-informed Domain Adaptation Paradigm for Cross-machine Transfer Diagnosis](https://doi.org/10.1016/j.knosys.2024.111499)" proposed by He et al.


## Citation
```
@article{chen2025deep,
  title={Deep discriminative domain adaptation network considering sampling frequency for cross-domain mechanical fault diagnosis},
  author={Chen, Guiping and Xiang, Dong and Liu, Tingting and Xu, Feng and Fang, Ke},
  journal={Expert Systems with Applications},
  pages={127296},
  year={2025},
  publisher={Elsevier}
}
```

 
