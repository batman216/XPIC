#pragma once
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"

template <typename PIC, typename MPI>
struct Input {

  using idx_type = PIC::index_type;
  using val_type = PIC::value_type;
  using Const    = PIC::Constant;
  using Container = typename PIC::ParticleContainer;
  static constexpr idx_type n_species = PIC::n_species, 
                                   dim_x = PIC::dim_x, 
                                   dim_v = PIC::dim_v;

  idx_type ng1;

  std::array<val_type,dim_x> L_gl, h_gl, a_gl, b_gl;
  std::array<val_type,dim_x> L, h, a, b;
  std::array<idx_type,dim_x> ncell_gl;
  std::array<idx_type,dim_x> nl, nr, ng_in, ng, ncell;
  std::array<idx_type,n_species> np_per_cell, np;
  std::array<val_type,n_species> n0, q, m, weight;

  std::size_t stopat, out_interval_1, out_interval_2;

  std::size_t out_ef, out_bf, out_jf;
  std::array<std::size_t,n_species> out_px, out_pv, out_pef;

  using json = nlohmann::json;

  val_type time_factor, dt;

  Input(const MPI& mpi) {
    
    std::ifstream input_file("x_input.json");
    json input = json::parse(input_file);

    for (int d=0; d<dim_x; ++d)
      ncell_gl[d] = input["#cell"][d];
  
    for (int s=0; s<n_species; ++s) {
      np_per_cell[s] = input["#particle per cell"][s];
      m[s]           = input["particle mass"][s];
      q[s]           = input["particle charge"][s];
      n0[s]          = input["particle number density"][s];
      out_px[s]      = input["print particle x"][s];
      out_pv[s]      = input["print particle v"][s];
      out_pef[s]     = input["print efield on particle"][s];

    }
      
    dt = input["time factor"];

    out_interval_1 = input["large output interval"];
    out_interval_2 = input["small output interval"];

    out_ef = input["print efield on cell"];
    out_bf = input["print bfield on cell"];
    out_jf = input["print jfield on cell"];

    stopat = input["run steps"];
    
    a_gl[0]     = 0.0;
    b_gl[0]     = static_cast<val_type>(ncell_gl[0]);

    L[0] = static_cast<val_type>(ncell_gl[0]) / (PIC::DVD ? mpi.world_size : 1);
    a[0] = PIC::DVD ? L[0]*mpi.world_rank : a_gl[0];
    b[0] = a[0] + L[0];
    nr[0] = 2; nl[0] = 2;
    
    ncell[0] = ncell_gl[0]/ (PIC::DVD ? mpi.world_size : 1);
    ng_in[0] = ncell[0] + 1;
    ng[0]    = ng_in[0] + nr[0] + nl[0];
    ng1      = ng[0];

    for (int s=0; s<n_species; ++s) {
      np[s] = ncell_gl[0]*np_per_cell[s]/mpi.world_size;
      weight[s] = n0[s] / np_per_cell[s];
    }
    
  }


};
