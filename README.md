
# Diffusion playground
A place to play with Diffusion models

## About the code
### Two ddpm versions:
The differences vs ddpm-v1 and ddpm-v2 are minor. 
- v-2 precompute some scalars like alpha-bars and square-roots of other scalars beforehand and therefore the code is shorter. 
- v1 on the other hand compute some of these scalars on the fly and is a bit longer but more straight forward and pedagogic

### Denoiser architecture

I added two architecture:
- Dummy is just stacked MLPs that map an image to image of the same dimension
- GenericUnet receives a list of spacial dimensions and creates a Unet with that many intermediate layers, each in the according spatial dimension. For example GenericUnet(scales=(32, 16, 8)) will have 6 intermediate feature maps **64x64->16x16->8x8->16x16->64x64**.

## Usage
Have a look at all the possible parameters at main.py

### Train on mnist/cifar
```
python3 main.py
```
Outputs after 9000 train steps:

<p float="center">
  <img src="readme_images/mnist_step-9000.png" width="200"/>
  <img src="readme_images/mnist_losses.png" width="260" /> 
</p>

```
python3 main.py --dataset cifar -c 3 --num_epochs 1000
```
Outputs after 70000 train steps:

<p float="center">
  <img src="readme_images/cifar_step-70000.png" width="200"/>
  <img src="readme_images/cifar_losses.png" width="260" /> 
</p>

### Train on other dataset
dataset folder should have sub-folders with images
```
python3 main.py --dataset <path to dataset> --c 3
```
Results on FFHQ 64x64 after 222000 steps:

<p float="center">
  <img src="readme_images/FFHQ64-222000.png" width="200"/>
  <img src="readme_images/FFHQ64-losses.png" width="260" /> 
</p>
# Credits
- https://github.com/BrianPulfer/PapersReimplementations/blob/main/ddpm/models.py
- https://github.com/VSehwag/minimal-diffusion.git
- https://github.com/cloneofsimo/minDiffusion
