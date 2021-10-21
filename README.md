# Boost Your Own Depth

## Apply our monocular depth boosting to your own network!

Our new [Google Colaboratory notebook](./colab/byod.ipynb) is now available.  [October 2021]   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/compphoto/BoostYourOwnDepth/blob/main/colab/byod.ipynb)

You can mix'n'match depths from different networks...:
|RGB | Base and details from [MiDaS][1] | Base from [MiDaS][1] and details from [LeRes][2]|
|----|------------|-----------|
|![patchselection](./figures/dts_rgb.jpg)|![Patchexpand](./figures/dts_midas.png)|![Patchexpand](./figures/dts_mix.png)|


...or use edited depths for improvements:
|RGB | Base and details from [MiDaS][1] | With edited base from [MiDaS][1]|
|----|------------|-----------|
|![patchselection](./figures/lunch_rgb.jpg)|![Patchexpand](./figures/lunch_orig.png)|![Patchexpand](./figures/lunch_edited.png)|



For more information on this project:
### Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging 

> S. Mahdi H. Miangoleh\*, Sebastian Dille\*, Long Mai, Sylvain Paris, Yağız Aksoy.
> [Main pdf](http://yaksoy.github.io/papers/CVPR21-HighResDepth.pdf),
> [Supplementary pdf](http://yaksoy.github.io/papers/CVPR21-HighResDepth-Supp.pdf),
> [Project Page](http://yaksoy.github.io/highresdepth/).
> [Github repo](https://github.com/compphoto/BoostingMonocularDepth).

[![video](./figures/video_thumbnail.jpg)](https://www.youtube.com/watch?v=lDeI17pHlqo)

## Citation

This implementation is provided for academic use only. Please cite our paper if you use this code or any of the models.
```
@INPROCEEDINGS{Miangoleh2021Boosting,
author={S. Mahdi H. Miangoleh and Sebastian Dille and Long Mai and Sylvain Paris and Ya\u{g}{\i}z Aksoy},
title={Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging},
journal={Proc. CVPR},
year={2021},
}
```

## Credits

The "Merge model" code skeleton (./pix2pix folder) was adapted from the [pytorch-CycleGAN-and-pix2pix][3] repository.\
[1]: https://github.com/intel-isl/MiDaS/tree/v2\
[2]: https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS\
[3]: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
