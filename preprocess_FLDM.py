import os
import os.path as osp
from os import listdir as osls
import zipfile
import shutil
import numpy as np
import scipy as sp
import tensorflow as tf
from PIL import Image
from tqdm import tqdm 
import PIL 
import trimesh 
import sys


import utils.io_utils as mio
import utils.utils as utils 
import utils.dataset as ds 

MAX_VERTS = 80000
TARGET_VERTS = 30000
MAX_FACES = 2 * TARGET_VERTS + 100

K_NORMALS = 30 # Number of neighbors for normal estiamtion
K_GEO = 32 # Number of neighbors for geometry estimation
K_UPSAMPLE = 8 # Number of neighbors for upsampling


NLEVELS = 2 # Number of levels in UNET
FACTOR = 2 # Downsampling factor
NUM_RING_SAMPLES = 64 # Texture samples per one ring -- 64/128 reccomended. Should be greater than or equal to average number of pixels per one ring.
MIN_TEX_DIM = 1024 # Downsample/upsample texture to have this as its minimal spatial dimension

uvw = utils.regular_samples(20)

MAX_PIX = uvw.shape[0] * MAX_FACES 


SPLITS = [("test", 16, 1), ("train", 100, 5)]


###########################
###### CHANGE THESE #######
###########################

MESH_FILE = "" # Location of .obj mesh file with texture coordinates per triangle
TEXTURE_FILE = "" # Location of .png/.jpg texture file
SAVE_PATH = "" # Path to save output .tfrecords files


print(" Processing mesh...", flush=True)


V, F, C = mio.load_mesh(MESH_FILE)
T = mio.load_png(TEXTURE_FILE)

TH = T.shape[0]
TW = T.shape[1]

min_dim = min(TH, TW)

TH = int(round(MIN_TEX_DIM * (TH / float(min_dim))))
TW = int(round(MIN_TEX_DIM * (TW / float(min_dim))))

tex_dim = np.asarray([TH, TW], dtype=np.int32)

if min_dim < MIN_TEX_DIM:
  TR = np.array(Image.fromarray(T).resize((TW, TH), resample=PIL.Image.Resampling.BILINEAR)).astype(np.uint8)
else:
  TR = np.array(Image.fromarray(T).resize((TW, TH), resample=PIL.Image.Resampling.LANCZOS)).astype(np.uint8)
  
VR, FR = utils.uniform_remesh(np.copy(V), np.copy(F), MAX_VERTS)

VR, FR = utils.get_largest_component(VR, FR)

print("Got components...", flush=True)
# Get pixel indices in image of texture map 
I, J = utils.get_valid_tex_image_ind(F, TR, C)

TRV = np.zeros_like(TR)
TRV[I, J, :] = TR[I, J, :]
TR = TRV 

#mio.save_png(TR, geom_path + 'valid_tex.jpg' )

print("Num verts in largest conn. component: {}".format(VR.shape[0]), flush=True)

# Compute per-vertex bases 
temp_geom = {}
temp_geom = utils.get_conv_geometry(temp_geom, VR, FR, nLevels=1, factor=1, 
                    k_geo=K_GEO,  k_upsample=K_UPSAMPLE,
                    k_normals=100)

bases = temp_geom["bases_0"]

for (split, NUM_MESHES, NUM_SHARDS) in SPLITS:
  
  split_path = os.path.join(SAVE_PATH, split)
  
  if not osp.exists(split_path):
    os.makedirs(split_path)
 
  for q in range(NUM_SHARDS):
    record_file = split_path + 'geom_{}.tfrecords'.format(q)
    writer = tf.io.TFRecordWriter(record_file)
    
    for k in tqdm(range(NUM_MESHES)):
      
      found_tri = False 
      count = 0 
      while(found_tri == False):
        
        try:

          hind, VS, FS = utils.fps_remesh(VR, FR, bases, TARGET_VERTS)
          
          
          geom = {} 
          geom["vert_map"] = hind 
          geom["nodes"] = VS 
          
          #print("Getting bases geometry...", flush=True)
          bases_S = bases[hind, ...]
          geom = utils.get_bases_geometry(geom, VS, FS, bases_S, nLevels=NLEVELS, factor=FACTOR, 
                            k_geo=K_GEO,  k_upsample=K_UPSAMPLE,
                            k_normals=K_NORMALS) 
          

          #print("got bases geometry", flush=True)
          # Compute tex pix mappings 
          tri_coords = utils.get_remeshed_tri_coords(V, F, C, uvw, VS, FS)
          
          
          pix_vals, pix_faces, pix_bary, valid_ind = utils.tex_pix_to_mesh_alt(FS, TR, uvw, tri_coords, I, J)

          pix_logs = utils.compute_bary_logs(FS, VS, bases_S, pix_faces, pix_bary)

          pix_tris = FS[pix_faces, :]
          
          pts = np.sum(VS[pix_tris, :] * pix_bary[..., None], axis=1)
          IJ = np.concatenate((I[..., None], J[..., None]), axis=-1)
          
          ring_logs, ring_ind, _, _ = utils.get_ring_samples(VS, FS, bases_S, pts, IJ, NUM_RING_SAMPLES)

          ring_vals = TR[ring_ind[..., 0], ring_ind[..., 1], :].astype(np.float32) / 255.0    
          
          found_tri = True 
          
        
        except:
          count = count + 1
          found_tri = False
          if (count >= 5):
            print("Taking more than five tries to find valid subsampled triangulation...", flush=True)
         
      
      # Normalize 
      for l in range(NLEVELS):
        
        mass = geom["hei_mass_{}".format(l)]
        
        unit_mass = np.mean(mass)
        log_scale = 1.0 / np.sqrt( unit_mass / np.pi)
        
        geom["hei_mass_{}".format(l)] = mass / unit_mass 
        
        geom["logs_{}".format(l)] = geom["logs_{}".format(l)] * log_scale 
        
        if (l == 0):
          pix_logs = pix_logs * log_scale 
          ring_logs = ring_logs * log_scale


      geom["pix_vals"] = pix_vals
      geom["valid_ind"] = valid_ind
      geom["pix_tris"] = pix_tris
      geom["pix_bary"] = pix_bary
      geom["pix_logs"] = pix_logs
      geom["I"] = I 
      geom["J"] = J 
      geom["tex_dim"] = tex_dim

      geom["ring_logs"] = ring_logs 
      geom["ring_vals"] = ring_vals 
    
      #geom["nn_map"] = nn_ind 
    
      
      # Encode and write 
      ex, key_list = ds.encode_example(geom)
      
      
      if (k == 0):
        mio.save_dict(split_path + "shape_keys.json", key_list) 
        mio.save_png(TR, SAVE_PATH + 'valid_tex.jpg' )

      writer.write(ex)
      
      
    # Close after processing each shape 
    writer.close() 
print("Done...", flush=True)



