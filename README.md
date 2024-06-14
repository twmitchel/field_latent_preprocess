# [Single Mesh Diffusion Models with Field Latents for Texture Generation, CVPR 2024](https://single-mesh-diffusion.github.io/): Pre-processing
### Pre-processing routines for Field Latent VAEs and Field Latent Diffusion Models complimenting the [official code release](https://github.com/google-research/google-research/tree/master/mesh_diffusion)

## Installation
CMake and Eigen are required for installation.

Clone this repository and its submodules 
```
$ git clone --recurse-submodules https://github.com/twmitchel/field_latent_preprocess.git
```
Afterwards, run
'''
pip install -r requirements.txt
'''
in the cloned directory.

## FL-VAE Pre-processing
The script `preprocess_FLVAE.py` generates train + test splits of planar triangulations for pre-training the FL-VAE on images as described in the paper. The data is stored as `.tfrecords` files.  Set the `SAVE_PATH` variable on line `20` of the file to the path of the directory where you want to store the data. Afterwards, run `python3 preprocess_FLVAE.py` to generate the data.

## FLDM Pre-processing
The script `preprocess_FLDM.py` generates train + test splits for training the diffusion models on a single textured mesh as described in the paper. The data is stored as `.tfrecords` files. Set the `MESH_FILE` variable on line `46` to be the location of the `.obj` mesh file which includes texture uv coordinates per triangle. Set the `TEXTURE_FILE` variable to the on line `47` to be the location of the `.png` or `.jpg` texture file. Set `SAVE_PATH` on line `48` to the path of the directory where you want to store the data. Afterwards, run `python3 preprocess_FLDM.py` to generate the data. 

Note that these `.tfrecords` files can be quite large (up to 100 GB) so choose a suitable location accordingly. The file size can be decreased by setting the `TARGET_VERTS` variable on line `21` to a lower number (say 10K) though this will probably decrease the quality of the generated textures and the FL-VAE should be pre-trained with a larger latent dimension and with coarser triangulations.
