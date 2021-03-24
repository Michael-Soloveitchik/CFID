<img src='images/faces_full (1).png' align="center" width=500>

<br><br><br><br>

# CFID
[Project Page](https://michael-soloveitchik.github.io/CFID/) |  [Paper](https://arxiv.org/abs/2103.11521)


TensorFlow2 implementation of Conditional Frechet Inception Distance metric. For example, given a low-resolution (LR) image, ground-truth high-resolution image (HR) and some super-resolution model (SR). The CFID metric is able to measure the exact distance between HR and SR given LR. In comparison to classic Frechet Inception Distance (FID), CFID considers the input LR image. It measureד the similarity between HR and SR regarding the input image. Unlike FID, CFID requires paired (LR,HR) data for comparison.

**Note**: The current software works well with TensorFllow 2.4.0

<!-- <img src='imgs/teaser.jpg' width=850>   -->
**Conditional Frechet Inception Distance.**  
[Michael Soloveitchik](https://new.huji.ac.il/people/%D7%9E%D7%99%D7%9B%D7%90%D7%9C-%D7%A1%D7%95%D7%9C%D7%95%D7%91%D7%99%D7%99%D7%A6%D7%99%D7%A7/),
 [Tzvi Diskin](https://new.huji.ac.il/people/%D7%A6%D7%91%D7%99-%D7%93%D7%99%D7%A1%D7%A7%D7%99%D7%9F/), [Efrat Morin](https://en.earth.huji.ac.il/people/efrat-morin/), [Ami Wiesel](https://www.cs.huji.ac.il/~amiw/).  
 The Hebrew University of jerusalem, 2021.

## Example results
<img src='imgs/results_matrix.jpg' width=820>  


## Other Implementations
- [[Tensorflow]](https://github.com) by __.
- [[Tensorflow]](https://github.com) by __.

## Prerequisites
- Windows, Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN


## Getting Started ###
### Installation
- Clone this repo:
```bash
git clone -b master --single-branch https://github.com/Michael-Soloveitchik/CFID.git
cd CFID
```
- Install TensorFllow and dependencies from https://www.tensorflow.org/ 

For pip users:
```bash
bash ./scripts/install_pip.sh
```

For conda users:
```bash
bash ./scripts/install_conda.sh
```


### Generate Morphing Videos
- We can also produce a morphing video similar to this [GIF](imgs/day2night.gif) and Youtube [video](http://www.youtube.com/watch?v=JvGysD2EFhw&t=2m21s).
```bash
bash ./scripts/video_edges2shoes.sh
```
Results can be found at `./videos/edges2shoes/`.


### Metrics

Figure 6 shows realism vs diversity of our method.

- **Realism** We use the Amazon Mechanical Turk (AMT) Real vs Fake test from [this repository](https://github.com/phillipi/AMT_Real_vs_Fake), first introduced in [this work](http://richzhang.github.io/colorization/).

- **Diversity** For each input image, we produce 20 translations by randomly sampling 20 `z` vectors. We compute LPIPS distance between consecutive pairs to get 19 paired distances. You can compute this by putting the 20 images into a directory and using [this script](https://github.com/richzhang/PerceptualSimilarity/blob/master/compute_dists_pair.py) (note that we used version 0.0 rather than default 0.1, so use flag `-v 0.0`). This is done for 100 input images. This results in 1900 total distances (100 images X 19 paired distances each), which are averaged together. A larger number means higher diversity.

### Citation

If you find this useful for your research, please use the following.

```
@inproceedings{zhu2017toward,
  title={Toward multimodal image-to-image translation},
  author={Zhu, Jun-Yan and Zhang, Richard and Pathak, Deepak and Darrell, Trevor and Efros, Alexei A and Wang, Oliver and Shechtman, Eli},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```
### Acknowledgements

This code is written by us.
