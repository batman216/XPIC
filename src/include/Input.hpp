
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
  

  val_type time_factor, dt;

  Input(const MPI& mpi) {
    
    time_factor = 0.2;

    m[0] = 1;    q[0] = -1;
    m[1] = 1836; q[1] = 1;
    //m[2] = 1836; q[2] = 1;
    np_per_cell[0] = 1200;
    np_per_cell[1] = 1200;
    //np_per_cell[2] = 1200;

    n0[0] = 1;
    n0[1] = 1;
    //n0[2] = 1;

    a_gl[0]     = 0;
    L_gl[0]     = 1200;
    b_gl[0]     = a_gl[0] + L_gl[0];
    ncell_gl[0] = 6000;
    h[0]        = L_gl[0]/ncell_gl[0];

    L[0] = L_gl[0] / (PIC::DVD ? mpi.world_size : 1);
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
    
    dt = h[0]/Const::c*time_factor;
  }


};
