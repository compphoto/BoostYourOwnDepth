# BoostYourOwnDepth

## Apply our monocular depth boosting to your own network!

Our [Google Colaboratory notebook](./colab/byod.ipynb) is now available.  [October 2021]   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/compphoto/BoostYourOwnDepth/blob/main/colab/byod.ipynb)

Follow the steps to easily boost your own depth.

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

The "Merge model" code skeleton (./pix2pix folder) was adapted from the [pytorch-CycleGAN-and-pix2pix][1] repository. 
[1]: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix