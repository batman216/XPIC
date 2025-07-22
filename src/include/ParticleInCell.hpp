
#include <cuda_fp16.h>

template <typename PIC>
struct Particle {

  using Container  = typename PIC::ParticleContainer;
  using idx_type   = PIC::index_type;
  using val_type   = PIC::value_type;
  static constexpr idx_type dim_x = PIC::dim_x, 
                               dim_v = PIC::dim_v;

  std::array<Container,dim_x> x;
  std::array<Container,dim_v> v;
  std::array<Container,dim_v> E,B;

  idx_type np;
  val_type w, q, m, qdm;
};


template <typename PIC>
struct Cell {
  using Container = typename PIC::CellContainer;
  using idx_type   = PIC::index_type;
  using val_type   = PIC::value_type;
  static constexpr idx_type n_species = PIC::n_species, 
                                dim_x = PIC::dim_x, 
                                dim_v = PIC::dim_v;
  // scalar fields
  Container eden/*charge density*/, epot/* electric potential*/;
  std::array<Container,n_species> edens/*charge density of each species*/;
  std::array<val_type,dim_x> L, h, a, b, L_gl, a_gl, b_gl;

  idx_type ng1;
  val_type dt;
  std::array<idx_type,dim_x> ng_al, ng_in, nl, nr;
  // vector fields
  std::array<Container,dim_v> efield, bfield, jfield;
  std::array<std::array<Container,dim_v>,n_species> jsfield;

};

template <typename PIC>
struct Species {

  using idx_type = PIC::index_type;
  using Particle = PIC::Particle;
  static constexpr idx_type n_species = PIC::n_species;
  std::array<Particle,n_species> species;
  
  Particle& operator[](idx_type s) {
    return species[s];
  }

  const Particle& operator[](idx_type s) const {
    return species[s];
  }

};


template <typename PIC> 
struct Constant {

  static constexpr PIC::value_type c = 7.14;

};

template <typename idx_type>
constexpr idx_type static_pow(idx_type val,int n) {
  return  (n==0) ? 1 : (val*static_pow(val,n-1));
}


struct MPIWrapper;
template <typename PIC, typename MPI>
struct Input;

template <typename idx_type, typename val_type, 
          template<typename> typename _ParticleContainer,
          template<typename> typename _CellContainer,
          bool     _DVD,  /*  */
          idx_type _n_species, 
          idx_type _dim_x, 
          idx_type _dim_v,
          idx_type _interp_order>
struct ParticleInCell {

  using index_type = idx_type;
  using value_type = val_type;
  using weight_type = val_type;
  using ParticleContainer = _ParticleContainer<val_type>;
  using CellContainer     = _CellContainer<val_type>;
  using Species           = Species<ParticleInCell>;
  using Particle          = Particle<ParticleInCell>;
  using Cell              = Cell<ParticleInCell>;
  using Input             = Input<ParticleInCell,MPIWrapper>;
  using Constant          = Constant<ParticleInCell>;
  static constexpr idx_type n_species = _n_species, dim_x = _dim_x, dim_v = _dim_v, interp_order = _interp_order;
  static constexpr idx_type n_node = static_pow(interp_order*2,dim_x);
  static constexpr bool DVD = _DVD;
  
};

