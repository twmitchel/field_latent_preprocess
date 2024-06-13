#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <queue>
#include <stdexcept>
#include <iostream>
#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h> 

namespace py = pybind11;

#include "Eigen/Dense"
#include "Eigen/Sparse"



struct 
{
  bool operator()(std::pair<int, double> a, std::pair<int, double> b) const { return a.second < b.second; }
} geo_less;




std::vector<std::vector<int>> get_conn_neighbors(const Eigen::MatrixXd& V, const std::vector<std::unordered_set<int>>& rings, const std::vector<int>& row_indices, const std::vector<int>& col_indices, int sample_factor=1, int k_neigh=128)
{
  Eigen::VectorXi mask = Eigen::VectorXi::Constant(V.rows(), -1);

  for (int l = 0; l < col_indices.size(); l++)
  {
    mask(col_indices[l]) = l;
  }

  std::vector<std::vector<int>> neigh; 
  neigh.resize(row_indices.size(), std::vector<int>{});

  std::vector<int> iso;


  for (int l = 0; l < row_indices.size(); l++)
  {
    std::unordered_map<int, double> dists;
    std::unordered_map<int, double> dists_mask; 
    
    dists[row_indices[l]] = 0.0;

    if (mask(row_indices[l]) >= 0)
    {
      dists_mask[mask(row_indices[l])] = 0.0;
    }
    
    std::unordered_set<int> shell = rings[row_indices[l]];
    
    
    bool got_nearest = false;

    int count = 0;
    double prev_min_k = -1.0;
    int num_loops = 0;
    while(got_nearest == false)
    {
      std::unordered_set<int> new_shell;

      for (const auto& n : shell)
      {
        double min_dist = std::numeric_limits<double>::max();

        for (const auto& rn: rings[n])
        {
          if (dists.find(rn) != dists.end())
          {
            double d = (V.block(rn, 0, 1, 3) - V.block(n, 0, 1, 3)).norm() + dists[rn];

            if (d < min_dist)
            {
              min_dist = d;
            }
          }
          else if (shell.find(rn) == shell.end()) //(seen.find(rn) == seen.end())
          {
            new_shell.insert(rn);
          }
        }

        if (dists.find(n) != dists.end())
        {
          std::cout << "DUPLICATE NEIGHBORS IN REFINED, nLoops = " << num_loops + 1 << std::endl;
          exit(0);
        }
        dists[n] = min_dist;

        if (mask(n) >= 0)
        {
          if (dists_mask.find(mask(n)) != dists_mask.end())
          {
            std::cout << "DUPLICATE NEIGHBORS" << std::endl;
            std::cout << "node " << mask(row_indices[l]) << ", neighbor " << mask(n) << std::endl;
            exit(0);
          }
          dists_mask[mask(n)] = min_dist; 
        }
      }

      shell = new_shell; 

      if (dists_mask.size() >= k_neigh)
      {
        std::vector<std::pair<int, double>> dists_mask_vec(dists_mask.begin(), dists_mask.end());

        std::sort(dists_mask_vec.begin(), dists_mask_vec.end(), geo_less);

        double min_k = dists_mask_vec[k_neigh-1].second;

        if (prev_min_k < 0)
        {
          prev_min_k = min_k;
        }
        else if (prev_min_k > min_k)
        {
          count = 0;
          prev_min_k = min_k;
        }
        else
        {
          count += 1;

          if (min_k != prev_min_k)
          {
            std::cout << "Min K = " << min_k << ", prev min k = " << prev_min_k << std::endl;
            exit(0);
          }
        }
      
        if (count == sample_factor || shell.size() == 0)
        {
          got_nearest = true;

          //if (mask[row_indices(l)] != 0 && dists_mask[0].first != mask)

          for (int j = 0; j < k_neigh; j++)
          {
            neigh[l].push_back(dists_mask_vec[j].first);
          }
        }

      }

      if (shell.size() == 0 && got_nearest == false)
      {                
        ///std::cout << "ISOLATED VERTEX" << std::endl;
        ///exit(0);

        throw std::runtime_error("ISOLATED_VERTEX");
      }

      num_loops = num_loops + 1; 
    }

  }

  return neigh;

}


std::vector<std::vector<std::vector<int>>> get_heiarchy_conn(const Eigen::MatrixXd& V, const std::vector<std::unordered_set<int>>& rings, const std::vector<std::vector<int>>& heiarchy, int k_neigh=128)
{
  std::vector<std::vector<std::vector <int>>> conn; 

  for (int l = 0; l < heiarchy.size(); l++)
  {
    int scale_factor = std::pow(2, l);

    conn.push_back(get_conn_neighbors(V, rings, heiarchy[l], heiarchy[l], scale_factor, k_neigh));
  }

  return conn;
}


////////////////////////////////////////////////////////////////////////////////
/////////////// From: https://github.com/nmwsharp/geometry-central /////////////
////////////////////////////////////////////////////////////////////////////////

double plane_angle(const Eigen::Vector3d& u, const Eigen::Vector3d& v, const Eigen::Vector3d& normal)
{
  // Put u in plane with the normal
  Eigen::Vector3d N = normal.normalized();
  Eigen::Vector3d u_plane = (u - u.dot(N) * N).normalized();
  Eigen::Vector3d basis_Y = normal.cross(u_plane).normalized();

  double x_comp = v.dot(u_plane);
  double y_comp = v.dot(basis_Y);

  return std::atan2(y_comp, x_comp);
}

Eigen::Vector3d rotate_around(const Eigen::Vector3d& v, const Eigen::Vector3d& axis, double theta)
{
  Eigen::Vector3d axis_n = axis.normalized();
  Eigen::Vector3d parallel_comp = axis_n * v.dot(axis_n);
  Eigen::Vector3d tangent_comp = v - parallel_comp;

  if (tangent_comp.squaredNorm() > 1e-8)
  {
    Eigen::Vector3d basis_X = tangent_comp.normalized();
    Eigen::Vector3d basis_Y = axis_n.cross(basis_X);
    double tangent_mag = tangent_comp.norm();

    Eigen::Vector3d rotated_v = tangent_mag * (std::cos(theta) * basis_X + std::sin(theta) * basis_Y);
    return rotated_v + parallel_comp;
  }
  else
  {
    return parallel_comp;
  }
}

// Compute the rotation such that v_source * r = v_target in the respective tangent bases. Requires normals and
// tangent bases have been computed.
std::tuple<Eigen::Vector2d, bool> transport_between_oriented(size_t source, size_t target, const Eigen::MatrixXd& basis_X, const Eigen::MatrixXd& basis_Y, const Eigen::MatrixXd& basis_Z)
{
  Eigen::Vector3d source_N = basis_Z.row(source);
  Eigen::Vector3d source_X = basis_X.row(source);

  Eigen::Vector3d target_N = basis_Z.row(target);
  Eigen::Vector3d target_X = basis_X.row(target);
  Eigen::Vector3d target_Y = basis_Y.row(target);

    // Flip orientation
  bool inverted = false;
  if (source_N.dot(target_N) < 0) 
  {
    target_N = -1.0 * target_N;
    target_Y = -1.0 * target_Y;
    inverted = true;
  }
  
  Eigen::Vector3d axis = target_N.cross(source_N);

  if (axis.norm() > 1e-6)
  {
    axis = axis.normalized();
  }
  else
  {
    axis = source_X;
  }

  double angle = plane_angle(source_N, target_N, axis);
  
  Eigen::Vector3d source_X_in_target_3 = rotate_around(source_X, axis, angle);
  Eigen::Vector2d source_X_in_target;
  source_X_in_target(0) = source_X_in_target_3.dot(target_X);
  source_X_in_target(1) = source_X_in_target_3.dot(target_Y);

  return std::make_tuple(source_X_in_target, inverted);
}


///////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// Geodesic FPS /////////////////////////////////////////
//// From https://github.com/rubenwiersma/deltaconv/blob/master/deltaconv/cpp/sampling.cpp ////
///////////////////////////////////////////////////////////////////////////////////////////////

// Data structure for Priority Queue
struct VertexPair {
	int		vId;
	double	distance;
	bool	operator> (const VertexPair &ref) const { return distance > ref.distance; }
	bool	operator< (const VertexPair &ref) const { return distance < ref.distance; }
};

// Computes the shortest distance between two points using Dijkstra's algorithm.
void computeDijkstra(const Eigen::MatrixXd& V, const std::vector<std::unordered_set<int>>& neigh, int source, Eigen::VectorXd &D) 
{

  std::priority_queue<VertexPair, std::vector<VertexPair>, std::greater<VertexPair>> DistanceQueue;

	D(source) = 0.0;
	VertexPair vp{ source, D(source) };
	DistanceQueue.push(vp);

	while (!DistanceQueue.empty())
  {
		VertexPair start = DistanceQueue.top();
		DistanceQueue.pop();

    for (const auto& vNeigh: neigh[source])
    {
      double dist, distTemp; 
      dist = (V.block(start.vId, 0, 1, 3) - V.block(vNeigh, 0, 1, 3)).norm();
      distTemp = start.distance + dist; 

      if (distTemp < D(vNeigh))
      {
        D(vNeigh) = distTemp;
        VertexPair adj{vNeigh, distTemp};
        DistanceQueue.push(adj);
      }

    }

	}

}


// Farthest point sampling on a point cloud, Euclidean distance
std::vector<int> constructFPS(const Eigen::MatrixXd& V,  std::vector<int> mask_ind, const size_t numSamples) 
{

	std::vector<int> sampleIdx;
  sampleIdx.resize(numSamples);

	Eigen::VectorXd D(mask_ind.size());
	D.setConstant(std::numeric_limits<double>::infinity());

	// Will be used to obtain a seed for the random number engine.
	std::random_device rd;
	// Standard mersenne_twister_engine seeded with rd().
	std::mt19937 gen(rd());
	// From 0 to (number of points - 1).
	std::uniform_int_distribution<> dist(0, mask_ind.size() - 1);	
	// Pick a random point to start with.
	sampleIdx[0] = mask_ind[dist(gen)];

	for (size_t i = 1; i < numSamples; i++) 
  {
		// Update distances.
    double max_dist = 0.0; 
    size_t max_ind = 0;
    for (size_t j = 0; j < mask_ind.size(); j++)
    {
      double dist = (V.block(sampleIdx[i-1], 0, 1, 3) - V.block(mask_ind[j], 0, 1, 3)).norm();

      if (dist < D[j])
      {
        D[j] = dist;
      }
      
      if (D[j] > max_dist)
      {
        max_dist = D[j];
        max_ind = j;
      }
    }
    
    sampleIdx[i] = mask_ind[max_ind];

	}
	return sampleIdx;
}

std::vector<std::vector<int>> get_coarse_heiarchy(const Eigen::MatrixXd& V, const std::vector<std::unordered_set<int>>& neigh, int nLevels, int factor = 4)
{
  std::vector<std::vector<int>> heiarchy;

  std::vector<int> fine;
  for (int l = 0; l < V.rows(); l++) {fine.push_back(l);}

  heiarchy.push_back(fine);

  for (int l = 1; l < nLevels; l++)
  {
    int numSamples = V.rows() / (std::pow(factor, l));

    //heiarchy.push_back(constructGeodesicFPS(V, neigh, heiarchy[l-1], numSamples));
    heiarchy.push_back(constructFPS(V, heiarchy[l-1], numSamples));

  }

  return heiarchy;  
}

