import numpy as np
from math import pi as PI
import tensorflow as tf
import json
import glob
Glob = glob.glob
Open = open 


def load_dict(file_name):
  
  with Open(file_name) as json_file:
    data = json.load(json_file)
  
  return data 

def encode_example(geom):
  
  features = {}
  
  key_list = {}
  for k in geom:
    
    feat = geom[k]
    
    if (feat.dtype.char in np.typecodes["AllFloat"]):
      feat = tf.convert_to_tensor(feat.astype(np.float32), dtype=tf.float32)
      key_list[k] = ["float", geom[k].shape]
      
    else:
      if (k == "texture"):
        #print("Saving texture", flush=True)
        feat = tf.convert_to_tensor(feat.astype(np.uint8), dtype=tf.uint8)
        key_list[k] = ["uint8", geom[k].shape]
      else:
        feat = tf.convert_to_tensor(feat.astype(np.int32), dtype=tf.int32)
        key_list[k] = ["int", geom[k].shape]
      
    
    feat_serial = tf.io.serialize_tensor(feat)

    features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[feat_serial.numpy()]))
      
  return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString(), key_list

      
def encode_shape_example(geom):
  
  features = {}
  
  key_list = {}
  for k in geom:
    
    
    if (k == "num_levels"):
      features["num_levels"] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[geom["num_levels"]]))
      
      #key_list[k] = "int"
    else:
      feat = geom[k]
      
      if (feat.dtype.char in np.typecodes["AllFloat"]):
        feat = tf.convert_to_tensor(feat.astype(np.float32), dtype=tf.float32)
        key_list[k] = ["float", geom[k].shape]
        
      else:
        if (k == "texture"):
          feat = tf.convert_to_tensor(feat.astype(np.uint8), dtype=tf.uint8)
          key_list[k] = ["uint8", geom[k].shape]
        else:
          feat = tf.convert_to_tensor(feat.astype(np.int32), dtype=tf.int32)
          key_list[k] = ["int", geom[k].shape]
      
      feat_serial = tf.io.serialize_tensor(feat)
      

       
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[feat_serial.numpy()]))
      
  return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString(), key_list


def load_tfr_dataset(files, ordered=False):
  
  ignore_order = tf.data.Options()
  if not ordered:
    ignore_order.experimental_deterministic = False 
    
  dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
  dataset = dataset.with_options(ignore_order)

  
  return dataset


def load_shape_tfrecords(path):

  shape_keys = load_dict(path + "/shape_keys.json")

  files = Glob(path + "*.tfrecords")

  return load_tfr_dataset(files), shape_keys

def get_des(key_list):
  shape_des = {}
  
  for k in key_list:
    shape_des[k] = tf.io.FixedLenFeature([], tf.string)
  
  return shape_des
  
  
def parse_data(ex, key_list, shape_des):
    
  example = tf.io.parse_single_example(ex, shape_des)
  shape = {}

  for k in key_list:
    dat = example[k]
    
    if key_list[k][0] == "float":
      feat = tf.io.parse_tensor(dat, tf.float32)
      feat = tf.ensure_shape(feat, key_list[k][1])

    else:
      
      if (key_list[k][0] == "uint8"):
        feat = tf.io.parse_tensor(dat, tf.uint8)
      else:
        feat = tf.io.parse_tensor(dat, tf.int32)
      
      feat = tf.ensure_shape(feat, key_list[k][1])

    shape[k] = feat
    
  return shape

def parser(key_list):
  
  shape_des = get_des(key_list)
  
  return lambda ex : parse_data(ex, key_list, shape_des)



def get_shape_des(key_list):
  shape_des = {'num_levels': tf.io.FixedLenFeature([], tf.int64)}
  
  for k in key_list:
    shape_des[k] = tf.io.FixedLenFeature([], tf.string)
  
  return shape_des
  
  
def parse_shape_data(ex, key_list, shape_des):
    
  example = tf.io.parse_single_example(ex, shape_des)
  shape = {}
  shape["num_levels"] = tf.cast(example['num_levels'], tf.int64)

  for k in key_list:
    dat = example[k]
    
    if key_list[k][0] == "float":
      feat = tf.io.parse_tensor(dat, tf.float32)
      feat = tf.ensure_shape(feat, key_list[k][1])



    else:
      feat = tf.io.parse_tensor(dat, tf.int32)
      feat = tf.ensure_shape(feat, key_list[k][1])

    shape[k] = feat
    
  return shape

def shape_parser(key_list):
  
  shape_des = get_shape_des(key_list)
  
  return lambda ex : parse_shape_data(ex, key_list, shape_des)



