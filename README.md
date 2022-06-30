# Cross-Domain Style Mixing for Face Cartoonization

## Overview
This repository contains an PyTorch implementation of ["Cross-Domain Style Mixing for Face Cartoonization"](https://arxiv.org/abs/2205.12450)

## Table of Contents
- Getting Started
  - Dependencies and Installation
  - How Construct?

- Preparation
- Crawling Webtoon Character Data
- 

- Training
- Generation(CDSM)
- 

##Getting Started
### Dependencies and Installation
- NVIDIA GPU + CUDA CuDNN 
- Python 3

```
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch # Check Cuda version and install  proper version Torch
$ conda install Ninja  # must install conda ( easy way )
```

## Preparation ( Collecting Data & Training )
### Preparing Webtoon Dataset
Paper use crawled webtoon dataset and iCartoonFace, but this repo use only webtoon data. you can collect webtoon data from [bryandlee/naver-webtoon-data](https://github.com/bryandlee/naver-webtoon-data) ( you can change collected dataset resolution by SRCNN, I set 1024x1024 same with paper )

### Preparing Generator
Before do CDSM, you must prepare 2 stylegan2 generator. First is Pretrained Source Domain Stylegan2 Generator ( here is FFHQ pretrained ). Second is Fine-Tunned Target Domain Stylegan2 Generator. ( here is fine tunned by webtoon dataset) you can training and fine-tunning stylegan2 at [rosinality/stylegan2-pytorch ](rosinality/stylegan2-pytorch). 

### Preparing Restyle Encoder ( GAN Inversion )
We need 2 GAN Inversion Models( Encoder ) each source domain and target domain. you can training each Restyle Encoder from [yuval-alaluf/restyle-encoder](https://github.com/yuval-alaluf/restyle-encoder). ( paper use restlye encoder. ) We already have each domain stylegan2 generator. So we use these two generator for training each domain Restyle Encoder. ( `--stylegan_weights `) 

## Generation (CDSM)
This Repository follow overall code process from [yuval-alaluf/restyle-encoder](https://github.com/yuval-alaluf/restyle-encoder). Because CDSM Paper use Restyle Encoder and CDSM can be implemented in Generator Process. ( So I reconstruct `main.py,` `gan_inversion.py` and `modify models/stylegan2/model.py` `cross_forward `function. )

### Pretrained_Weights
| Path | Description
| :--- | :----------
| [FFHQ StyleGAN](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view)     | StyleGAN2 model trained on FFHQ with 1024x1024 output resolution.
| [Webtoon_StyleGAN]()     | StyleGAN2 model trained on Webtoon Dataset with 1024x1024 output resolution.
| [FFHQ - ReStyle + pSp](https://drive.google.com/file/d/1sw6I2lRIB0MpuJkpc8F5BJiSZrc0hjfE/view?usp=sharing)  | ReStyle applied over pSp trained on the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset.
| [Webtoon - ReStyle + pSp]()   | ReStyle applied over pSp trained on the Webtoon Dataset.

python new_main.py --exp_dir=./experiments --load_numpy



stylegan2 pytorch ffhq link **https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view**

0번으로 데이터 크롤링하는거 있는데 이것도 깃허브 체크해서 올려놔야겠다. 

1,2번 같은 경우에는 두개의 깃허브를 마킹하면서 여기에 따라서 학습을 하면된다. Cartoon dataset으로 이정도로할까?
Preparing이라고 하면될듯

저렇게 학습이 다 됐으면 마지막에 cross.py만 쓰면 된다는거. 체크해서 만들어두면 되겠다.
+ 참고하면 좋을 논문들은 꼭 다 링크걸어두고 간단히 설명 적어둬야겠다. trgb이해를 위해서 color distortion이해를 위해서 그 stylespace논문 appendix참고해라 등등.

쓸 생각을 하자. 순서를 먼저 적자. -> train인지 fine tunning인지 명확히 확인해둬야함. 같은 space로 보내는거 생각할 떄 encoder가 fine tunning이 확실했던 것 같긴함.
1. Stylegan2 Finetunning with Carttondataset ( with prepared pretrained ffhq dataset )
2. Training Restyle Encoder with Cartoon Dataset
3. inference로 넘어옴 Layer Swapped Generator를 준비
4. source image와 선택된 k개의 target이미지들을 gan inversion encoder(restyle)을 통해서 latent code를 구하고 target이미지들의 경우 평균을 낸다. 
5. 그렇게 한 뒤에 trgb replacement를 하고 style mixing을 하고, 미리 준비해놓은 layer swapped generator에서 trgb replacement와 style mixing을 한다. 
