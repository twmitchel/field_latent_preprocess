# This was about as fun to write as it is to read.
# Will try to clean this up eventually.

import numpy as np
import scipy as sp
import trimesh as trimesh 
import flbind as flb
import pyvista
import pyacvd
from heapdict import heapdict 
import igl
from sklearn.neighbors import NearestNeighbors
from math import pi as PI

EPS = 1.0e-8 


# Use farthest point sampling to create planar traingulation of square grid
def fps_grid(height, num_points):
  
  assert num_points <= height * height 
  
  ni, nj = np.meshgrid(np.arange(height), np.arange(height), indexing='ij')
  
  ni = np.reshape(ni, (-1, ))
  nj = np.reshape(nj, (-1, ))
  
  V = np.concatenate((ni[..., None], nj[..., None]), axis=-1)
  
  
  VG = np.concatenate((V, np.zeros_like(V[:, 0])[..., None]), axis=-1)
  
  samples = flb.get_fps(VG, num_points)
  
  VS = V[samples[:, 0], :]
  
  corners = np.asarray([[0, 0], [0, height-1], [height-1, 0], [height-1, height-1]], dtype=VS.dtype)
  
  VS = np.concatenate((corners, VS), axis=0)
  
  VS = np.unique(VS, axis=0)[:num_points, :]
  
  assert VS.shape[0] == num_points 
  
  F = sp.spatial.Delaunay(VS.astype(np.float32), qhull_options="QJ").simplices
  
  e1 = VS[F[:, 1], :] - VS[F[:, 0], :]
  e2 = VS[F[:, 2], :] - VS[F[:, 0], :]
  
  areas = 0.5 * np.abs(np.cross(e1, e2))
  
  valid_tri = np.flatnonzero(areas > 1.0e-6)
  
  F = F[valid_tri, :]
  
  return VS, F 

# Triangle areas of a mesh
def tri_area(V, F):
  E1 = V[F[:, 1], :] - V[F[:, 0], :]
  E2 = V[F[:, 2], :] - V[F[:, 0], :]
  
  n = np.cross(E1, E2)
  
  return 0.5 * np.linalg.norm(n, axis=-1)


def get_largest_component(V, F):
  
  # Get largest connected component
  C = igl.facet_components(F)
  
  comps, num_times = np.unique(C, return_counts=True)
  
  largest_comp = comps[np.argmax(num_times)]
  
  comp_inds = np.flatnonzero(C == largest_comp)
  
  F = F[comp_inds, :]
  
  # Renumber vertices 
  v_ind = np.unique(F)
  
  VN = V[v_ind, :]
  
  v_map = {}
  
  for l in range(v_ind.shape[0]):
    v_map[v_ind[l]] = l 
    
  FN = np.zeros_like(F)
  
  for l in range(F.shape[0]):
    for j in range(3): 
      
      FN[l, j] = v_map[F[l, j]]
  
  return VN, FN  
  
# Barycentric coordinates of a point p, in a triangle w/ vertices a, b, c
# p, a, b, c: N X 3 
def barycentric(p, a, b, c):
  
  v0 = b - a
  v1 = c - a
  v2 = p - a
  
  d00 = np.sum(v0 * v0, axis=-1)
  d01 = np.sum(v0 * v1, axis=-1)
  d11 = np.sum(v1 * v1, axis=-1)
  d20 = np.sum(v2 * v0, axis=-1)
  d21 = np.sum(v2 * v1, axis=-1)
  
  denom = d00 * d11 - d01 * d01
  
  mask = np.abs(denom) > EPS 
  
  denom = mask * denom  + (1.0 - mask) * np.sign(denom) * EPS 
  
  mask = np.abs(denom) > (EPS / 2.0)
  
  denom = mask * denom + (1.0 - mask) * EPS 
  
  
  v = (d11 * d20 - d01 * d21) / denom
  w = (d00 * d21 - d01 * d20) / denom
  u = 1.0 - v - w
  
  assert np.isnan(v).any() == False 
  assert np.isnan(u).any() == False 
  
  return u, v, w 

def regular_samples(n):
  
  ng = ( ( n + 1 ) * ( n + 2 ) ) // 2
  t = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
  
  tg = np.zeros( ( ng, 2 ) )

  p = 0

  for i in range( 0, n + 1 ):
    for j in range( 0, n + 1 - i ):

      tg[p,0] = ( float (     i     ) * t[0,0]   
                + float (         j ) * t[1,0]   
                + float ( n - i - j ) * t[2,0] ) / float ( n )

      tg[p,1] = ( float (     i     ) * t[0,1]   
                + float (         j ) * t[1,1]   
                + float ( n - i - j ) * t[2,1] ) / float ( n )
      p = p + 1
  
  v1 = np.asarray([0.0, 0.0, 0.0])
  v2 = np.asarray([1.0, 0.0, 0.0])
  v3 = np.asarray([0.0, 1.0, 0.0])
  
  bcoords = []
  
  for j in range(tg.shape[0]):
    
    pt = np.asarray([tg[j, 0], tg[j, 1], 0.0])
    
    a, b, c = barycentric(pt, v1, v2, v3)
    
    bcoords.append([a, b, c])
    
  return np.asarray(bcoords, dtype=np.float32)
  
def get_grid_pixel_bary(V, F, height):
  
  I = np.zeros(height*height, dtype=np.int32)
  J = np.zeros(height*height, dtype=np.int32)
  
  pc = 0
  for i in range(height):
    for j in range(height):
      I[pc] = i 
      J[pc] = j 
      pc = pc + 1 
        
  V_in = np.concatenate((V, np.zeros_like(V[:,0, None])), axis=-1).astype(np.float32)
  P = np.concatenate((I[..., None], J[..., None], np.zeros_like(I)[..., None]), axis=-1).astype(np.float32)
  
  pix_faces, pix_bary, p_logs, valid_ind, find = get_nearest_points_on_mesh(V_in, F, P)
  
  return pix_faces, pix_bary, p_logs, valid_ind, I, J, find 
   
  
def get_nearest_points_on_mesh(V, F, P): 
  
  mesh = trimesh.Trimesh(vertices=V, faces=F) 
  
  C , dist, find = trimesh.proximity.closest_point(mesh, P)
  
  p_faces = F[find, :]
  
  p_tris = V[p_faces, :]
    
  u, v, w = barycentric(C, p_tris[:, 0, :], p_tris[:, 1, :], p_tris[:, 2, :])
  
  p_bary = np.concatenate((u[..., None], v[..., None], w[..., None]), axis=-1)
  
  min_bc = np.min(p_bary)
  max_bc = np.max(p_bary)
  
  if (min_bc < -1.0e-3 or max_bc > 1.0 + 1.0e-3):
    print(" OOB min, max bary = {}, {}".format(min_bc, max_bc), flush=True)
  
  
  valid_ind = np.less(dist, 1e-1).astype(np.float32)
  
  # Compute "logs"
  
  p_logs = C[:, None, :] - p_tris
  p_logs = p_logs[..., :2]
  
  return p_faces, p_bary, p_logs, valid_ind, find

def get_ring_samples(V, F, B, points, IJ, num_samples): 
  
  areas = tri_area(V, F)
  
  # Compute one-rings + tri areas    
  rings = {} 
  
  for l in range(F.shape[0]):
  
    for j in range(3):
      
      if F[l, j] not in rings:
        s = set()
      else:
        s = rings[F[l, j]]
      
      s.add(l)
      
      rings[F[l, j]] = s 


  logs = np.zeros((V.shape[0], num_samples, 2), dtype=V.dtype)
  bary = np.zeros((V.shape[0], num_samples, 3), dtype=V.dtype)
  tri = np.zeros((V.shape[0], num_samples), dtype=np.int32)
  emb = np.zeros((V.shape[0], num_samples, 3), dtype=V.dtype)
  
  # For each vertex uniformly sample points in one-ring
  # https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle-in-3d
  for l in range(V.shape[0]):
    
    r_inds = np.asarray(list(rings[l]))
    
    r_areas = areas[r_inds]
    
    if np.sum(r_areas) > 1.0e-18:
      
      r_areas = r_areas / np.sum(r_areas)
    else:
      r_areas = np.ones_like(r_areas)
      
    s_tris = np.random.choice(r_inds, num_samples, p=r_areas)
    
    r1 = np.sqrt(np.random.uniform(size=(num_samples, ), low=1.0e-2, high=0.99))
    r2 = np.random.uniform(size=(num_samples, ), low=1.0e-2, high=0.99)
    
    u = 1.0 - r1 
    v = r1 * (1.0 - r2) 
    w = r2 * r1  


    pts = V[F[s_tris, 0], :] * u[:, None] + V[F[s_tris, 1], :] * v[:, None] + V[F[s_tris, 2], :] * w[:, None]
    pts[0, :] = V[l, :] 
    
    emb[l, ...] = pts 
    
    dir_vec = pts - V[l, None, :]
    dir_mag = np.linalg.norm(dir_vec, axis=-1, keepdims=True)
    
    dir_vec = np.matmul(B[l, None, ...], dir_vec[..., None])[..., 0]
    
    dir_vec = dir_vec / (np.linalg.norm(dir_vec, axis=-1, keepdims=True) + 1.0e-6)
    
    logs[l, ...] = dir_mag * dir_vec 
    
    tri[l, ...] = s_tris 
    
    bary[l, ...] = np.concatenate((u[..., None], v[..., None], w[..., None]), axis=-1)
    
  
  emb = np.reshape(emb, (-1, 3))
  
  nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(points.astype(np.float32))
  _, nind = nbrs.kneighbors(emb.astype(np.float32))
  
  s_IJ = np.reshape(IJ[nind, :], (V.shape[0], num_samples, 2))
  
  return logs, s_IJ, tri, bary

def get_conn(V, F):
  
  # Compute one-rings + tri areas    
  areas = tri_area(V, F)

  rings = {} 
  mass = np.zeros(V.shape[0], dtype=np.float32)

  for l in range(F.shape[0]):
    
    for j in range(3):
      
      mass[F[l, j]] = mass[F[l, j]] + (1.0 / 3.0) * areas[l]
      
      if F[l, j] not in rings:
        s = set()
      else:
        s = rings[F[l, j]]
      
      for k in range(3):
        s.add(F[l, k])
                
      rings[F[l, j]] = s 
  
  conn = []
  
  for l in range(V.shape[0]):
    
    for n in rings[l]:
      if (n != l):
        conn.append([l, n])
  
  conn = np.asarray(conn, dtype=np.int32)
  
  return conn, mass
    
    
def get_mesh_orientation(vertices, faces):


  num_vertices = vertices.shape[0]
  num_faces = faces.shape[0]

  direc = np.zeros((num_vertices, 3), dtype=vertices.dtype)

  axis_1 = vertices[faces[:, 1], :] - vertices[faces[:, 0], :]
  axis_2 = vertices[faces[:, 2], :] - vertices[faces[:, 0], :]

  normals_faces = np.cross(axis_1, axis_2)

  for l in range(num_faces):
    for j in range(3):
      direc[faces[l, j], :] = direc[faces[l, j], :] + normals_faces[l, :]


  return direc 

#transport rows <- cols 
#           j   <-  i
def compute_transport(V, B, rows, cols):
  
  dir_i_j = np.matmul(B[cols, ...], (V[rows, :] - V[cols, :])[..., None])[..., 0]
  dir_j_i = np.matmul(B[rows, ...], (V[cols, :] - V[rows, :])[..., None])[..., 0]
  
  # Check for zero_indices 
  xport = np.zeros( (rows.shape[0], ), dtype=V.dtype)
  
  adj_ind = np.flatnonzero(rows != cols)
  
  phi_i_j = np.arctan2(dir_i_j[adj_ind, 1], dir_i_j[adj_ind, 0])
  phi_j_i = np.arctan2(dir_j_i[adj_ind, 1], dir_j_i[adj_ind, 0])
  
  xport[adj_ind] = (phi_j_i + PI) - phi_i_j 
  
  return xport 

def get_normals(vertices, neigh, direc=None, consistent=True):
  """

  Args:
    vertices:
    edges:
    direc:
    consistent:

  Returns:

  """


  num_vertices = vertices.shape[0]
  dim = vertices.shape[1]
  k_neigh = neigh.shape[1]
  
  edges = np.reshape(neigh, (num_vertices * k_neigh, ))
  normals = np.zeros((num_vertices, dim), dtype=vertices.dtype)

  '''
  y_matrix = np.reshape(vertices[edges[:, 1], :] - vertices[edges[:, 0], :],
                        (num_vertices, -1, dim))
  '''
  y_matrix = vertices[:, None, :] - np.reshape(vertices[edges, :], (num_vertices, k_neigh, 3))
  
  basis_3d, _, _ = np.linalg.svd(
      np.matmul(np.transpose(y_matrix, (0, 2, 1)), y_matrix))

  normals = basis_3d[..., -1]

  normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

  # If direction at vertices is specified, flip normals
  # if they point in opposite direction
  if direc is not None:
    angle_cos = np.sum(normals * direc, axis=1)

    flip_ind = np.flatnonzero(angle_cos < 0)

    normals[flip_ind, :] = -1.0 * normals[flip_ind, :]
    #normals[flip_ind, :] *= -1.0

  # Ensure normals point in consistent directions
  # if consistency flag is set
  if consistent and (direc is None):
    rows = edges[:, 0]
    cols = edges[:, 1]

    keep_ind = (rows != cols).nonzero()

    rows = rows[keep_ind]
    cols = cols[keep_ind]

    weights = np.ones(
        rows.shape[0], dtype=vertices.dtype) - np.abs(
            np.sum(normals[rows, :] * normals[cols, :], axis=1))

    span = sp.sparse.csgraph.minimum_spanning_tree(
        sp.sparse.csr_matrix((weights, (rows, cols)),
                             shape=(num_vertices, num_vertices))).tocoo()

    row_ind = span.row
    col_ind = span.col

    span_neigh = []

    for l in range(num_vertices):
      span_neigh.append([])

    for l in range(row_ind.shape[0]):
      span_neigh[row_ind[l]].append(col_ind[l])
      span_neigh[col_ind[l]].append(row_ind[l])

    visited = np.empty(num_vertices, dtype=np.bool_)
    visited.fill(False)

    init_ind = 0
    cache = [init_ind]
    visited[init_ind] = True

    while cache:

      ind = cache.pop()

      for l in range(len(span_neigh[ind])):

        neigh_ind = span_neigh[ind][l]

        if not visited[neigh_ind]:

          cache.append(neigh_ind)
          visited[neigh_ind] = True

          if np.sum(normals[ind, :] * normals[neigh_ind, :]) < 0:
            normals[neigh_ind, :] = -1.0 * normals[neigh_ind, :]

  return normals


def nudged_copysign(a, b):
  
  return np.sign( (np.sign(a) + 0.5) * (np.sign(b) + 0.5) )

def get_tangent_bases(vertices, neigh, direc=None):


  normals = get_normals(vertices, neigh, direc=direc, consistent=False)
  
  num_vertices = normals.shape[0]
  dim = normals.shape[1]

  ## Generate unit orthogonal vector
  ## From: https://math.stackexchange.com/questions/137362/how-to-find
  ## -perpendicular-vector-to-another-vector
  sxz = nudged_copysign(normals[:, 0], normals[:, 2])
  syz = nudged_copysign(normals[:, 1], normals[:, 2])
  
  x_0 = normals[:, 2] * sxz 
  x_1 = normals[:, 2] * syz
  x_2 = -1.0 * (normals[:, 0] * sxz + normals[:, 1] * syz)
  
  axis_x = np.concatenate((x_0[..., None], x_1[..., None], x_2[..., None]),
                          axis=-1).astype(normals.dtype)

  axis_x = np.cross(normals, axis_x)
  
  axis_x = axis_x / np.linalg.norm(axis_x, axis=-1, keepdims=True)

  axis_y = np.cross(normals, axis_x)

  axis_y = axis_y / np.linalg.norm(axis_y, axis=-1, keepdims=True)

  basis = np.zeros((num_vertices, dim - 1, dim), dtype=normals.dtype)

  basis[:, 0, :] = axis_x
  basis[:, 1, :] = axis_y

  return basis

def get_neigh_logs_xport(V, bases, neigh):
  
  num_verts = neigh.shape[0]
  k_neigh = neigh.shape[-1]
  
  source = np.repeat(np.arange(neigh.shape[0])[:, None], k_neigh, axis=1)
  
  source = np.reshape(source, (-1,))
  neigh = np.reshape(neigh, (-1,))

  xport  = np.reshape(compute_transport(V, bases, source, neigh), (num_verts, k_neigh)) 
  
  # Compute logs
  pt = V[source, :] - V[neigh, :]
  
  dist = np.linalg.norm(pt, axis=-1)
  
  pos = np.matmul(bases[neigh, ...], pt[..., None])[..., 0]
  
  logs = pos * (dist / (np.linalg.norm(pos, axis=-1) + 1.0e-6))[..., None]
  
  logs = np.reshape(logs, (num_verts, k_neigh, 2))
  
  return logs, xport

def get_pool_xport(V, bases, fine_ind, coarse_ind, fine_adj):
  
  num_verts = fine_ind.shape[0]
  k_neigh = fine_adj.shape[1]
  num_edges = num_verts * k_neigh 
  
  cols = coarse_ind[np.reshape(fine_adj, (num_verts * k_neigh, ))]
    
  rows = np.reshape(np.tile(fine_ind[:, None], (1, k_neigh)), (num_edges, ))
  
  
  # get_transport(i, j): i <- j 
  xport_unpool = np.reshape(compute_transport(V, bases, rows, cols), (num_verts, k_neigh))
  xport_pool = np.reshape(compute_transport(V, bases, cols, rows), (num_verts, k_neigh))
  
  return np.concatenate((xport_pool[..., None], xport_unpool[..., None]), axis=-1)
    
def get_conv_geometry(geom, V, F, nLevels, factor, 
                      k_geo,  k_upsample,
                      k_normals, grid=False):

  num_verts = V.shape[0]
  
  #conn, mass, spec_freq, spec_bases = get_cotan_spectra(V, F)
  conn, mass = get_conn(V, F)
  
  ## Compute robust quantities
  heiarchy, heiarchy_levels, heiarchy_neigh, heiarchy_mass, heiarchy_adj = flb.get_irregular_geometry(V, conn, mass, nLevels=nLevels, factor=factor, k_geo=max(k_geo, k_normals), k_upsample=k_upsample)
  

  direc = get_mesh_orientation(V, F)
  bases_neigh = heiarchy_neigh[np.flatnonzero(heiarchy_levels == 0), :k_normals]
  
  if grid:
    bases = np.zeros((V.shape[0], 2, 3), dtype=V.dtype)
    bases[:, 0, 0] = 1.0
    bases[:, 1, 1] = 1.0
  else:
    bases = get_tangent_bases(V, bases_neigh, direc=direc)
    
  
  ########################################
  #### Unpack everything and zero-pad ####
  ########################################
  
  for l in range(nLevels):
    
    level_ind = np.flatnonzero(heiarchy_levels == l)
    
    # Heiarchy indices
    hei_l = heiarchy[level_ind, 0]
    
    # Heiarchy neighbors (num_h x k_geo)
    hei_neigh_l = heiarchy_neigh[level_ind, :]
    
    # Heiarchy mass values
    hei_mass_l = heiarchy_mass[level_ind, 0]
        
    bases_l = bases[hei_l, ...]

    logs_l, xport_l = get_neigh_logs_xport(V[hei_l, :], bases_l, hei_neigh_l[:, :k_geo])

    ## Pooling adjacency and weights

    if (l > 0):
      up_adj_l = heiarchy_adj[(l-1)*num_verts:l*num_verts, :]
      hei_prev = heiarchy[np.flatnonzero(heiarchy_levels == l-1), 0]
      unpool_adj_l = up_adj_l[hei_prev, :]
      unpool_vals_l = get_pool_xport(V, bases, hei_prev, hei_l, unpool_adj_l)
 
    
    ##########################################################
    ##################### Zero-Padding #######################
    ##########################################################
    
    geom["hei_{}".format(l)] = hei_l
    geom["hei_neigh_{}".format(l)] = hei_neigh_l 
    geom["hei_mass_{}".format(l)] = hei_mass_l 
    geom["bases_{}".format(l)] = bases_l 
    
    
    geom["logs_{}".format(l)] = logs_l 
    geom["xport_{}".format(l)] = xport_l 
    
    ## Unpooling maps 
    if (l > 0):
      geom["unpool_adj_{}_{}".format(l-1, l)] = unpool_adj_l 
      geom["unpool_vals_{}_{}".format(l-1, l)] = unpool_vals_l
  
  return geom

def get_bases_geometry(geom, V, F, bases, nLevels, factor, 
                      k_geo,  k_upsample,
                      k_normals, grid=False):

  num_verts = V.shape[0]
  
  conn, mass = get_conn(V, F)
  
  ## Compute robust quantities
  heiarchy, heiarchy_levels, heiarchy_neigh, heiarchy_mass, heiarchy_adj = flb.get_irregular_geometry(V, conn, mass, nLevels=nLevels, factor=factor, k_geo=max(k_geo, k_normals), k_upsample=k_upsample)
  
  
  ########################################
  #### Unpack everything and zero-pad ####

  for l in range(nLevels):
    
    level_ind = np.flatnonzero(heiarchy_levels == l)
    
    # Heiarchy indices
    hei_l = heiarchy[level_ind, 0]
    
    # Heiarchy neighbors (num_h x k_geo)
    hei_neigh_l = heiarchy_neigh[level_ind, :]
    
    # Heiarchy mass values
    hei_mass_l = heiarchy_mass[level_ind, 0]
        
    bases_l = bases[hei_l, ...]

    logs_l, xport_l = get_neigh_logs_xport(V[hei_l, :], bases_l, hei_neigh_l[:, :k_geo])

    
    ## Pooling adjacency and weights

    if (l > 0):
      up_adj_l = heiarchy_adj[(l-1)*num_verts:l*num_verts, :]
      hei_prev = heiarchy[np.flatnonzero(heiarchy_levels == l-1), 0]
      unpool_adj_l = up_adj_l[hei_prev, :]
      unpool_vals_l = get_pool_xport(V, bases, hei_prev, hei_l, unpool_adj_l)
 
    
    ##########################################################
    ##################### Zero-Padding #######################
    ##########################################################
    
    geom["hei_{}".format(l)] = hei_l
    geom["hei_neigh_{}".format(l)] = hei_neigh_l 
    geom["hei_mass_{}".format(l)] = hei_mass_l 
    geom["bases_{}".format(l)] = bases_l 
    
    geom["logs_{}".format(l)] = logs_l 
    geom["xport_{}".format(l)] = xport_l 
    
    ## Unpooling maps 
    if (l > 0):
      geom["unpool_adj_{}_{}".format(l-1, l)] = unpool_adj_l 
      geom["unpool_vals_{}_{}".format(l-1, l)] = unpool_vals_l
  
  return geom 
   
def get_valid_tex_image_ind(F, T, C, eps_pix=30.0):
  
  # Get UV mesh
  CV = np.unique(np.reshape(C, (-1, 2)), axis=0)
  
  CF = np.zeros( (C.shape[0], 3), dtype=np.int32)
  
  for l in range(C.shape[0]):
    for j in range(3):
      cv_ind = np.argmin(np.linalg.norm(C[None, l, j, :] - CV, axis=-1))
      CF[l, j] = cv_ind 
      
  CV = np.concatenate((CV, np.zeros((CV.shape[0], 1), dtype=CV.dtype)), axis=-1)
  

  H = T.shape[0]
  W = T.shape[1]
  
  I, J = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
  I = np.reshape(I, (-1, ))
  J = np.reshape(J, (-1, ))
  
  V = 1.0 - I / (H - 1.0)
  U = J / (W - 1.0)
  
  UV = np.concatenate((U[..., None], V[..., None], np.zeros((V.shape[0], 1), dtype=V.dtype)), axis=-1)
  
  eps = eps_pix / W 
  
  if CV.shape[0] < 60000:
    
    mesh = trimesh.Trimesh(vertices=CV, faces=CF)

    _, dist, _ = trimesh.proximity.closest_point(mesh, UV)
  
  else:
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(CV)
    
    dist, _ = nbrs.kneighbors(UV)
    
  valid_ind = np.flatnonzero(dist <= eps)
  
  I = I[valid_ind]
  J = J[valid_ind]
  
  return I, J 

def get_remeshed_tri_coords(V, F, C, uvw, VR, FR):
  
  S = (V[F[:, 0], None, :] * uvw[None, :, 0, None] + 
       V[F[:, 1], None, :] * uvw[None, :, 1, None] +
       V[F[:, 2], None, :] * uvw[None, :, 2, None]) 
  
  f_inds = np.repeat(np.arange(F.shape[0])[:, None], uvw.shape[0], axis=1)

  SR = (VR[FR[:, 0], None, :] * uvw[None, :, 0, None] + 
       VR[FR[:, 1], None, :] * uvw[None, :, 1, None] +
       VR[FR[:, 2], None, :] * uvw[None, :, 2, None])
  
  f_inds = np.reshape(f_inds, (-1, ))
  S = np.reshape(S, (-1, 3))
  SR = np.reshape(SR, (-1, 3))
  
  nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(S)
  
  _, nind = nbrs.kneighbors(SR)
  
  nind = np.squeeze(nind)
  f_inds = f_inds[nind]
  b0, b1, b2 = barycentric(SR, V[F[f_inds, 0], :], V[F[f_inds, 1], :], V[F[f_inds, 2], :])
  
  #b0 = np.reshape(b0, (FR.shape[0], uvw.shape[0]))
  #b1 = np.reshape(b1, (FR.shape[0], uvw.shape[0]))
  #b2 = np.reshape(b2, (FR.shape[0], uvw.shape[0]))
  
  
  tri_coords = (C[f_inds, 0, :] * b0[:, None] + 
                C[f_inds, 1, :] * b1[:, None] + 
                C[f_inds, 2, :] * b2[:, None]) 
  
  return tri_coords 

def compute_bary_logs(F, V, B, coord_faces, coord_bary):

  vi1 = F[coord_faces, 0]
  vi2 = F[coord_faces, 1]
  vi3 = F[coord_faces, 2]


  pt = V[vi1, :] * coord_bary[:, 0, None] + V[vi2, :] * coord_bary[:, 1, None] + V[vi3, :] * coord_bary[:, 2,  None]


  dir1 = pt - V[vi1, :]
  dir2 = pt - V[vi2, :]
  dir3 = pt - V[vi3, :]

  dist1 = np.linalg.norm(dir1, axis=-1)
  dist2 = np.linalg.norm(dir2, axis=-1)
  dist3 = np.linalg.norm(dir3, axis=-1)

  pos1 = np.matmul(B[vi1, ...], dir1[..., None])[..., 0]
  pos2 = np.matmul(B[vi2, ...], dir2[..., None])[..., 0]
  pos3 = np.matmul(B[vi3, ...], dir3[..., None])[..., 0]


  
  pos1 = pos1 * ( dist1 / (np.linalg.norm(pos1, axis=-1) + 1.0e-6) )[..., None]
  pos2 = pos2 * ( dist2 / (np.linalg.norm(pos2, axis=-1) + 1.0e-6) )[..., None]
  pos3 = pos3 * ( dist3 / (np.linalg.norm(pos3, axis=-1) + 1.0e-6) )[..., None]

  return np.concatenate( (pos1[:, None, :], pos2[:, None, :], pos3[:, None, :]), axis=1)

def tex_pix_to_mesh_alt(F, T, uvw, tri_coords, I, J):
  
  coord_faces= np.repeat(np.arange(F.shape[0])[:, None], uvw.shape[0], axis=1)
  coord_bary = np.repeat(uvw[None, ...], F.shape[0], axis=0)

  #print("got 1", flush=True)

  coord_faces = np.reshape(coord_faces, (-1, ))
  coord_bary = np.reshape(coord_bary, (-1, 3))
  
  nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(tri_coords)

  H = T.shape[0]
  W = T.shape[1]

  s_coords = []

  V = 1.0 - I / (H - 1.0)
  U = J / (W - 1.0)

  s_coords = np.concatenate((U[..., None], V[..., None]), axis=-1)

  _, cind = nbrs.kneighbors(s_coords)

  nn_ind = np.squeeze(cind)

  valid_ind = np.ones_like(nn_ind).astype(np.float32)
  
  seen = {}   
  for l in range(nn_ind.shape[0]):
    
    ni = nn_ind[l]
    
    if ni in seen: 
      s = seen[ni]
      s.add(l)
      
      for k in s:
        valid_ind[k] = 0.0 
    else:
      s = set()
      s.add(l)
      seen[ni] = s 

  
  # Flag repeated indices 
  #print("got before pix vals...", flush=True)

  pix_vals = T[I, J, ...]
  
  return pix_vals, coord_faces[nn_ind, ...], coord_bary[nn_ind, ...], valid_ind 
  
#######################################
###### Uniform Remeshing ##############
#######################################
def geodesic_nearest(points, V, F):
  # From https://mosco.github.io/geodesicknn/geodesic_knn.pdf
  
  #pts = set(points.tolist())
  pts = {}
  for l in range(points.shape[0]):
    pts[points[l]] = l 

  # Compute one-rings 
  rings = {}
  
  for l in range(F.shape[0]):
    
    for j in range(3):
      
      if F[l, j] in rings:
        s = rings[F[l, j]]
      else:
        s = set()
      
      s.add(F[l, (j+1)%3])
      s.add(F[l, (j+2)%3])

      rings[F[l, j]] = s 

  
  # Initalize priority queue 
  nn = {} 
  
  Q = heapdict()
  
  visited = set() 
  k = 1 
  
  
  for l in range(V.shape[0]):
    nn[l] = [] 
    if l in pts:
      Q[(l, l)] = 0.0 
  
  while bool(Q):
    seed_pair, dist = Q.popitem() 
    seed = seed_pair[0]
    v0 = seed_pair[1]
    
    visited.add(seed_pair)
    
    nn_list = nn[v0]

    if len(nn_list) < k:
      nn[v0] = nn_list + [seed]
      
      for n in rings[v0]:
        
        edge = (seed, n)
        if edge not in visited:  
          new_dist = dist + np.linalg.norm(V[n, :] - V[v0, :])

          if edge in Q:
            Q[edge] = min(new_dist, Q[edge])
          else:
            Q[edge] = new_dist 
  
  nearest = np.zeros(V.shape[0], dtype=np.int32)
  for l in range(V.shape[0]):    
    nearest[l] = pts[nn[l][0]]

      
  return nearest 
  
def cluster_remesh(points, V, F, B):
  
  num_points = points.shape[0] 
  
  VR = V[points, :]
  
  if (np.unique(VR, axis=0).shape[0] != VR.shape[0]):
    print("REPEATED VERTICES IN FPS", flush=True)
    print("NUM POINTS = {}".format(VR.shape[0]), flush=True)
    print("NUM UNIQUE = {}".format(np.unique(VR, axis=0).shape[0]), flush=True)
      
  #BR =  B[points, :]
  
  labels = geodesic_nearest(points, V, F)
  
  #labels = euclidean_nearest(points, V)
  tris = []
  seen = set() 
  
  for l in range(F.shape[0]):
    
    l1 = labels[F[l, 0]]
    l2 = labels[F[l, 1]]
    l3 = labels[F[l, 2]]
    
    if (l1 == l2 or l1 == l3 or l2 == l3):
      continue 
    
    t = [l1, l2, l3]
    ts = tuple(np.sort(t).tolist())
    
    if ts not in seen:
      tris.append(t)
      seen.add(ts)
  
  FR = np.asarray(tris).astype(np.int32)
  
  unique_fv = np.unique(FR)

  #print("adding verts...", flush=True)
  if (VR.shape[0] < unique_fv.shape[0]):  
    
    print("CRITICAL CLUSTER REMESH ERROR...", flush=True)
  
  elif (VR.shape[0] > unique_fv.shape[0]):
    
   add_verts = np.setdiff1d(np.arange(VR.shape[0]), unique_fv)
   
   if add_verts.shape[0] > 500:
    print("Adding {} vertices...".format(add_verts.shape[0]))
   
   for l in range(add_verts.shape[0]):     
      
    mesh = trimesh.Trimesh(vertices=VR, faces=FR)

    _, _, t_ind = trimesh.proximity.closest_point(mesh, VR[None, add_verts[l], :])
    
    t_ind = t_ind[0]
    c_tri = FR[t_ind, :]
    
    new_tris = np.asarray([[c_tri[0], c_tri[1], add_verts[l]],
                           [c_tri[1], c_tri[2], add_verts[l]],
                           [add_verts[l], c_tri[2], c_tri[0]]])
    
    keep_faces = np.setdiff1d(np.arange(FR.shape[0]), np.asarray([t_ind]))
    
    FR = np.concatenate((FR[keep_faces, :], new_tris), axis=0) 

  return VR, FR
  
def fps_remesh(V, F, B, num_points):
  
  
  nMaxIter = 20
  found = False
  count = 0;
  while not found:
    

    points = flb.get_fps(V, num_points)
    points = points[:, 0]

    
    VR, FR = cluster_remesh(points, V, F, B)
    
        
    if (VR.shape[0] == np.unique(FR).shape[0]):
      found = True 
      if count > 0:
        print(" Took {} iters...".format(count), flush=True)
    else:

      count = count + 1 
      if count == nMaxIter:
        assert False == True 

  
  return points, VR, FR 
  

def get_remeshed_tri_coords(V, F, C, uvw, VR, FR):
  
  S = (V[F[:, 0], None, :] * uvw[None, :, 0, None] + 
       V[F[:, 1], None, :] * uvw[None, :, 1, None] +
       V[F[:, 2], None, :] * uvw[None, :, 2, None]) 
  
  f_inds = np.repeat(np.arange(F.shape[0])[:, None], uvw.shape[0], axis=1)

  SR = (VR[FR[:, 0], None, :] * uvw[None, :, 0, None] + 
       VR[FR[:, 1], None, :] * uvw[None, :, 1, None] +
       VR[FR[:, 2], None, :] * uvw[None, :, 2, None])
  
  f_inds = np.reshape(f_inds, (-1, ))
  S = np.reshape(S, (-1, 3))
  SR = np.reshape(SR, (-1, 3))
  
  nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(S)
  
  _, nind = nbrs.kneighbors(SR)
  
  nind = np.squeeze(nind)
  f_inds = f_inds[nind]
  b0, b1, b2 = barycentric(SR, V[F[f_inds, 0], :], V[F[f_inds, 1], :], V[F[f_inds, 2], :])
  
  #b0 = np.reshape(b0, (FR.shape[0], uvw.shape[0]))
  #b1 = np.reshape(b1, (FR.shape[0], uvw.shape[0]))
  #b2 = np.reshape(b2, (FR.shape[0], uvw.shape[0]))
  
  
  tri_coords = (C[f_inds, 0, :] * b0[:, None] + 
                C[f_inds, 1, :] * b1[:, None] + 
                C[f_inds, 2, :] * b2[:, None]) 
  
  return tri_coords 
  

def uniform_remesh(V, F, target_verts):

  # Total area 
  tri_areas = tri_area(V, F)
  max_a = np.max(tri_areas)
  A = np.sum(tri_areas) / V.shape[0]
  
  rad = np.sqrt(A / PI)
  
  max_el = 8 * rad  
  
  V, F = split_edges(V, F, max_el)    
 
  #NV, NF = igl.remove_duplicates(V, F, 1.0e-4)
  NV, _, _, NF = igl.remove_duplicate_vertices(V, F, 1.0e-4)

 
  
  if (NV.shape[0] < V.shape[0]):
    print("Removed {} duplicate vertices".format(V.shape[0] - NV.shape[0]))
    
    tri_areas = tri_area(NV, NF)
    
    if max_a < np.max(tri_areas):
      print("CATASTROPIC DUPLICATE REMOVAL!", flush=True)
      print("Mean tri area is {}".format(np.mean(tri_areas)), flush=True)
      print("Max tri area is {}".format(np.max(tri_areas)), flush=True)
  
  V = NV 
  F = NF 
  
  facets = np.concatenate( (3*np.ones_like(F[:, 0])[:, None], F), axis=1)  
  facets = np.hstack(facets)

  mesh = pyvista.PolyData(V, facets)
  
  clus = pyacvd.Clustering(mesh)

  if V.shape[0] <= target_verts:
    num_div = int(np.ceil(np.log(target_verts / float(V.shape[0]))/np.log(4)))
    #print(num_div, flush=True)
    clus.subdivide(num_div)
    
  clus.cluster(target_verts)
  
  remesh = clus.create_mesh() 

  rfaces = remesh.faces
  rverts = remesh.points
  
  rfaces = np.reshape(rfaces, (remesh.n_faces, 4))[:, 1:]

  # Remove any superfluous vertices 
  u_verts = np.unique(rfaces)
  
  u_inds = {}
  
  for j in range(u_verts.shape[0]):
    u_inds[u_verts[j]] = j 
  
  rverts = rverts[u_verts, :]
  
  rfaces2 = np.zeros((rfaces.shape[0], 3), dtype=F.dtype)
  for l in range(rfaces.shape[0]):
    for j in range(3):
      rfaces2[l, j] = int(u_inds[rfaces[l, j]])
  
  VR = np.zeros((rverts.shape[0], 3), dtype=V.dtype)
  FR = np.zeros((rfaces.shape[0], 3), dtype=F.dtype)
  
  for l in range(VR.shape[0]):
    for j in range(3):
      VR[l, j] = float(rverts[l, j])
      
  for l in range(FR.shape[0]):
    for j in range(3):
      FR[l, j] = int(rfaces2[l, j])
  
  #VR, FR = collapse_small_tris(VR, FR, 1.0e-14)
    
  return VR, FR  
  

def split_edges(V, F, max_el):
  
  Q = heapdict()
 
  
  for l in range(F.shape[0]):
    for j in range(3):
      e = tuple(np.sort([F[l, j], F[l, (j+1)%3]]))
      if e not in Q:
        Q[e] = max_el / (np.linalg.norm(V[e[0], :] - V[e[1], :]) + EPS)
  
  satisfied = False 
  
  num_split = 0
  while not satisfied: 
    
    e, e_l = Q.popitem() 
    
    if e_l >= 1.0: 
      satisfied = True 
    else: 
      num_split = num_split + 1 
      
      f_inds = np.flatnonzero((np.sum(F == e[0], axis=1) + np.sum(F == e[1], axis=1)) >= 2)
      
      keep_faces = np.setdiff1d(np.arange(F.shape[0]), f_inds)

      new_tris = []
      
      new_vert = (V[e[0], :] + V[e[1], :]) / 2.0 
      V = np.concatenate((V, new_vert[None, :]), axis=0) 
      v_ind = V.shape[0] - 1 
      
      for l in range(f_inds.shape[0]):
        face = F[f_inds[l], :]
        
        found_edge = False 
        
        for j in range(3):
          ef = tuple(np.sort([face[j], face[(j+1)%3]]))
        
          if (e == ef):
            found_edge = True 
            
            v1 = face[j]
            v2 = face[(j+1)%3]
            v3 = face[(j+2)%3]
            
            new_tris.append([v1, v_ind, v3])
            new_tris.append([v2, v3, v_ind])
            
            e1 = tuple(np.sort([v1, v_ind]))
            e2 = tuple(np.sort([v_ind, v3]))
            e3 = tuple(np.sort([v_ind, v2]))
            
            Q[e1] = max_el / (np.linalg.norm(V[e1[0], :] - V[e1[1], :]) + EPS)
            Q[e2] = max_el / (np.linalg.norm(V[e2[0], :] - V[e2[1], :]) + EPS)
            Q[e3] = max_el / (np.linalg.norm(V[e3[0], :] - V[e3[1], :]) + EPS)
            
            break 
      
        if not found_edge:
          print("FATAL EDGE DIVISION FAILURE", flush=True)
            
      F = np.concatenate((F[keep_faces, :], np.asarray(new_tris)), axis=0)
  
  print("Split {} edges".format(num_split), flush=True)      
  
  return V, F
        
def collapse_small_tris(V, F, eps):
  
  diag_length = np.linalg.norm(np.max(V, axis=0) - np.min(V, axis=0))
  
  V = igl.collapse_small_triangles(V, F, eps/diag_length)
  
  unique_fv = np.unique(F)
  D = {}
  VN = np.zeros((unique_fv.shape[0], 3), dtype=V.dtype)
  FN = np.zeros_like(F)
  
  for l in range(unique_fv.shape[0]):
    D[unique_fv[l]] = l 
    VN[l, :] = V[unique_fv[l], :]

  for l in range(F.shape[0]):
    for j in range(3):
      FN[l, j] = D[F[l, j]]
  
  V = VN 
  F = FN
  
  return V, F 
  
