#include "geometry.inl"



std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXi>
get_functional_geometry(const Eigen::MatrixXd& V, const std::vector<std::unordered_set<int>>& one_rings, 
                        const std::vector<std::vector<int>>& heiarchy, const Eigen::MatrixXd& mass_fine,  int k_geo = 128, int k_upsample=5)
{

  int num_verts = V.rows();
  int nLevels = heiarchy.size();
  

  // Compute connectivity
  std::vector<std::vector<std::vector<int>>> conn_geo = get_heiarchy_conn(V, one_rings, heiarchy, k_geo);

  // Get pooling/unpooling support + mass matrices 
  std::vector<Eigen::MatrixXd> mass; // level (0, 1, 2, ..., n)
  std::vector<std::vector<std::vector<int>>> refine_adj; // levels (0->1, 1->2, ..., n-1 -> n)

  mass.push_back(mass_fine);

  for (int l = 1; l < nLevels; l++)
  {

    std::vector<std::vector<int>> fine_adj = get_conn_neighbors(V, one_rings, heiarchy[0], heiarchy[l], std::pow(2, l), k_upsample);
    
    Eigen::MatrixXd coarse_mass(heiarchy[l].size(), 1);
    coarse_mass.setZero();
  
    for (int j = 0; j < heiarchy[l-1].size(); j++)
    {
      int n = fine_adj[heiarchy[l-1][j]][0];
    
      coarse_mass(n, 0) += mass[l-1](j, 0);
    }
  
    mass.push_back(coarse_mass);
    refine_adj.push_back(fine_adj);
  }


  // Convert to output representations
  int num_nodes = 0;
  int num_pool = 0;
  for (int l = 0; l < heiarchy.size(); l++)
  {
    num_nodes += heiarchy[l].size();

    if (l < heiarchy.size() - 1)
    {
      num_pool += heiarchy[l].size();
    }
  }

  Eigen::MatrixXi hei(num_nodes,1);
  Eigen::MatrixXi hei_levels(num_nodes, 1); 
  Eigen::MatrixXi hei_neigh(num_nodes, k_geo);
  Eigen::MatrixXd hei_mass(num_nodes, 1);
  Eigen::MatrixXi hei_adj(num_verts * (nLevels-1), k_upsample);


  int ind = 0;
  for (int l = 0; l < nLevels; l++)
  {
    for (int j = 0; j < heiarchy[l].size(); j++)
    {
      //std::cout << "Writing level " << l << ", node " << j << ", index " << ind;
 
      int n_ind = heiarchy[l][j]; 

      hei(ind, 0) = n_ind;
      hei_levels(ind, 0) = l;
      hei_mass(ind, 0) = mass[l](j, 0);

      for (int k = 0; k < k_geo; k++)
      {
        hei_neigh(ind, k) = conn_geo[l][j][k];
      }
      ind += 1;
    }
  }

  ind = 0;
  for (int l = 0; l < nLevels-1; l++)
  {
    for (int j = 0; j < num_verts; j++)
    {
      for (int k = 0; k < k_upsample; k++)
      {
        hei_adj(ind, k) = refine_adj[l][j][k];
      }
      ind += 1;
    }
  }

  //geom.reset();
  //cloud.reset();
  return std::make_tuple(hei, hei_levels, hei_neigh, hei_mass, hei_adj);

}


std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXi>
get_irregular_geometry(const Eigen::MatrixXd& V, const Eigen::MatrixXi& conn, const Eigen::MatrixXd& mass_fine, int nLevels=1, int factor = 4, int k_geo = 128, int k_upsample=5)
{
  int num_verts = V.rows();
  
  // Unpack neighbors 
  std::vector<std::unordered_set<int>> one_rings;
  one_rings.resize(num_verts, std::unordered_set<int>{});

  for (int l = 0; l < conn.rows(); l++)
  {
    one_rings[conn(l, 0)].insert(conn(l, 1)); 
  }

  for (int l = 0; l < num_verts; l++)
  {
    if (one_rings[l].find(l) != one_rings[l].end())
    {
      one_rings[l].erase(l);
    }
  }

  // Compute sampling heiarchy
  std::vector<std::vector<int>> heiarchy = get_coarse_heiarchy(V, one_rings, nLevels, factor);

  return get_functional_geometry(V, one_rings, heiarchy, mass_fine, k_geo, k_upsample);

}





Eigen::MatrixXi get_fps(const Eigen::MatrixXd& V, int num_samples)
{
  std::vector<int> mask_ind;
  for (int l = 0; l < V.rows(); l++)
  {
    mask_ind.push_back(l);
  }
  
  std::vector<int> sample_ind = constructFPS(V, mask_ind, num_samples);
  
  Eigen::MatrixXi out(num_samples, 1);
  

  for (int l = 0; l < num_samples; l++)
  {
    out(l, 0) = sample_ind[l];
  }

  return out;
}


Eigen::MatrixXd get_transport(const Eigen::MatrixXi& rows, const Eigen::MatrixXi& cols, const Eigen::MatrixXd& basis_X, const Eigen::MatrixXd& basis_Y, const Eigen::MatrixXd& basis_Z)
{
  int num_edges = rows.rows(); 

  Eigen::MatrixXd xport_angle(num_edges, 2);

  for (int l = 0; l < num_edges; l++)
  {
      Eigen::Vector2d r_col_to_row;
      bool inverted;

      std::tie(r_col_to_row, inverted) = transport_between_oriented(cols(l, 0), rows(l, 0), basis_X, basis_Y, basis_Z);

      //xport_angle(l, 0) = std::atan2(r_col_to_row(1), r_col_to_row(2));
      xport_angle(l, 0) = std::atan2(r_col_to_row(1), r_col_to_row(0));
      if (inverted)
      {
        xport_angle(l, 1) = -1.0;
      }
      else
      {
        xport_angle(l, 1) = 1.0;
      }
  }

  return xport_angle;
}


        



/////////////////////////////////////////////////
////////////////// Bindings /////////////////////
///////////////////////////////////////////////// 
PYBIND11_MODULE(flbind, m) {
  m.doc() = "Bindings for FL-VAE and FLDM pre-processing.";
  

  m.def("get_irregular_geometry", &get_irregular_geometry, "Compute geometric quantities", 
      py::arg("V"), py::arg("conn"),  py::arg("mass_fine"),  py::arg("nLevels"), py::arg("factor"), py::arg("k_geo"), py::arg("k_upsample"));

  m.def("get_fps", &get_fps, "Compute Euclidean_fps", 
    py::arg("V"),  py::arg("num_samples"));
  
  m.def("get_transport", &get_transport, "Get local transport angle from col ind to row ind",
       py::arg("rows"), py::arg("cols"), py::arg("basis_X"), py::arg("basis_Y"), py::arg("basis_Z"));

  
}
