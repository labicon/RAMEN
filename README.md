<h1 align="center"><strong>RAMEN: Real-time Asynchronous Multi-agent Neural Implicit Mapping</strong></h1>

<p align="center">
	<a href="https://scholar.google.com/citations?user=4uQNsj8AAAAJ&hl=zh-CN">Hongrui Zhao</a>, 
	<a href="https://www.borisivanovic.com/">Boris Ivanovic</a>,
	<a href="https://negarmehr.com/">Negar Mehr</a>,
</p>

<div align="center">
	<a href='https://arxiv.org/abs/2502.19592'><img src='https://img.shields.io/badge/arXiv-2308.16246-b31b1b'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 	<a href='https://iconlab.negarmehr.com/RAMEN/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 	<!-- <a href='https://www.youtube.com/watch?v=psPvanfh7SA&feature=youtu.be'><img src='https://img.shields.io/badge/Youtube-Video-blue'></a> -->
</div>


## Installation

Our environment has been tested on Ubuntu 18.04 (CUDA 10.2 with RTX2080Ti) and Ubuntu 20.04(CUDA 10.2/11.3 with RTX2080Ti). Torch1.12.1 is recommended to reproduce the results.

Clone the repo and create conda environment

```shell
git clone --recurse-submodules git@github.com:ZikeYan/activeINR.git && cd activeINR

# create conda env
conda env create -f environment.yml
conda activate activeINR
```

Install pytorch by following the [instructions](https://pytorch.org/get-started/locally/). 

```shell
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102

pip install -e .
```



## Acknowledgement

Our code is partially based on [Co-SLAM](https://github.com/HengyiWang/Co-SLAM). We thank the authors for making these codes publicly available.

## Citation

```
@inproceedings{Zhao2025RSS
  title={RAMEN: Real-time Asynchronous Multi-agent Neural Implicit Mapping},
  author={Zhao, Hongrui and Ivanovic, Boris and Mehr, Negar},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2025}
}
```
