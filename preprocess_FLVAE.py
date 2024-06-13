import os
import os.path as osp
from os import listdir as osls
import zipfile
import shutil
import numpy as np
import scipy as sp
from tqdm import tqdm
import tensorflow as tf
from PIL import Image 
import trimesh 
import sys
import flbind as flb


import utils.io_utils as mio
import utils.utils as utils 
import utils.dataset as ds 

SAVE_PATH = "" #Set this to be the directory where you want the output .tfrecords files to be stored


WIDTH = 512 # Dimension of squre image grid
NUM_SAMPLES = 5000 # Number of vertices sampled on grid
TRI_SAMPLES = 20 
MAX_TRIS = 2 * NUM_SAMPLES + 200

NUM_MESHES = 100
NUM_SHARDS = 10

K_NORMALS = 8
K_GEO = 25
K_UPSAMPLE = 8
NLEVELS = 1
FACTOR = 2

NUM_RING_SAMPLES = 64


SPLITS = [ ('test', 16, 1), ('train', 100, 5)] #Split, num meshes, num shards


NUM_RECORDS = 10


for (split, NUM_MESHES, NUM_RECORDS) in SPLITS:

  save_path = os.path.join(SAVE_PATH, split)
  
  os.makedirs(save_path, exist_ok=True)

  for s in range(NUM_RECORDS):

    record_file = save_path + 'im_meshes_W{}_V{}_TS{}_NM{}_{}.tfrecords'.format(WIDTH, NUM_SAMPLES, TRI_SAMPLES, NUM_MESHES, s)

    writer = tf.io.TFRecordWriter(record_file)
    

    for l in tqdm(range(NUM_MESHES)):
      
      geom = {} 

      V, F = utils.fps_grid(WIDTH, NUM_SAMPLES)
      
      pix_faces, pix_bary, pix_logs, valid_mask, I, J, _ = utils.get_grid_pixel_bary(V, F, WIDTH)
      
      
      geom["nodes"] = V 
      geom["pix_faces"] = pix_faces 
      geom["pix_bary"] = pix_bary
      geom["pix_logs"] = pix_logs 
      geom["valid_mask"] = valid_mask 
      geom["I"] = I 
      geom["J"] = J 

      
      # Compute conv geometry 
      VG = np.zeros((V.shape[0], 3), dtype=np.double)
      VG[:, 0] = V[:, 0].astype(np.double)
      VG[:, 1] = V[:, 1].astype(np.double)
      
          
      
      geom = utils.get_conv_geometry(geom, VG, F, NLEVELS, factor=FACTOR, 
                          k_geo=K_GEO, k_upsample=K_UPSAMPLE, 
                          k_normals=K_NORMALS, grid=True)
      
      pts = np.sum(VG[pix_faces, :] * pix_bary[..., None], axis=1)
      IJ = np.concatenate((I[..., None], J[..., None]), axis=-1)
      B = geom["bases_0"]
      
      ring_logs, ring_ind, _, _ = utils.get_ring_samples(VG, F, B, pts, IJ, NUM_RING_SAMPLES)

      geom["ring_logs"] = ring_logs 
      geom["ring_ind"] = ring_ind 
      
      
      ## Encode example 
      ex, key_list = ds.encode_example(geom)
        
      if (l == 0 and s == 0):
        mio.save_dict(save_path + "shape_keys.json", key_list) 
        key_template = geom.keys() 
        

      writer.write(ex)


    writer.close() 



