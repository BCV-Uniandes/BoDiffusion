# BoDiffusion

This is the official implementation of the paper: BoDiffusion: Diffusing Sparse Observations for Full-Body Human Motion Synthesis.<br>

## Paper
[BoDiffusion: Diffusing Sparse Observations for Full-Body Human Motion Synthesis](https://arxiv.org/pdf/2304.11118.pdf) <br/>
[Angela Castillo](https://angelacast135.github.io)<sup> 1*</sup>, [María Escobar](https://mc-escobar11.github.io)<sup> 1*</sup>, [Guillaume Jeanneret](https://guillaumejs2403.github.io)<sup> 2</sup>, [Albert Pumarola](https://www.albertpumarola.com)<sup> 3</sup>, [Pablo Arbeláez](https://scholar.google.com.co/citations?user=k0nZO90AAAAJ&hl=en)<sup>1</sup>, [Ali Thabet](https://scholar.google.com/citations?user=7T0CPEkAAAAJ&hl=en)<sup> 3</sup>, [Artsiom Sanakoyeu](https://gdude.de)<sup> 3</sup> <br/>
<sup>*</sup>Equal contribution.<br/>
<sup>1 </sup>Center for Research and Formation in Artificial Intelligence ([CinfonIA](https://cinfonia.uniandes.edu.co)), Universidad de Los Andes. <br/>
<sup>2 </sup>University of Caen Normandie, ENSICAEN, CNRS, France. <br/>
<sup>3 </sup>Meta AI. <br/>
<br/>

![](./videos/BoDiffusion_final1.mp4)

## Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch == 1.13.0  TorchVision == 0.15.2](https://pytorch.org/)
- NVIDIA GPU + [CUDA v10.1.243](https://developer.nvidia.com/cuda-downloads)

1. Clone repo

    ```bash
    git clone https://github.com/BCV-Uniandes/BoDiffusion
    ```

1. Install dependent packages

    ```bash
    cd BoDiffusion
    conda env create -f env.yaml
    ```


## Dataset Preparation

- Please refer to [this repo](https://github.com/eth-siplab/AvatarPoser#datasets) for details about the dataset organization and split.

## Train

- **Training command**: 

    ```bash
    python train.py
    ```


## Citations

If BoDiffusion helps your research, please consider citing us.<br>

``` latex
@inproceedings{castillo2023bodiffusion,
  title={BoDiffusion: Diffusing Sparse Observations for Full-Body Human Motion Synthesis},
  author={Castillo, Angela and Escobar, Maria and Jeannerete, Guillaume and Pumarola, Albert and Arbel{\'a}ez, Pablo and Thabet, Ali and Sanakoyeu, Artsiom},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

Find other resources in our [webpage](https://bcv-uniandes.github.io/bodiffusion-wp/).

## License and Acknowledgement

This project borrows heavily from [Guided Diffusion](https://github.com/openai/guided-diffusion), we thank the authors for their contributions to the community.<br>

## Contact

If you have any question, please email `a.castillo13@uniandes.edu.co`.