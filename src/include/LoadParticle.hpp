#include <random>
#include <thrust/generate.h>
#include <thrust/host_vector.h>


template <typename PIC, typename Input>
struct LoadParticle {

  Input in;

  LoadParticle(const Input& in) : in(in) {}

  using Container = typename PIC::ParticleContainer;
  using val_type  = typename PIC::value_type;
  static constexpr std::size_t n_species = PIC::n_species, 
                                   dim_x = PIC::dim_x, 
                                   dim_v = PIC::dim_v;
 
  template <typename Species>
  void operator()(Species& pa) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<val_type> udist(in.a[0],in.b[0]);
    //std::uniform_real_distribution<val_type> gdist(1,1.2);
    std::normal_distribution<val_type> gdist(0.0,0.1);

    thrust::host_vector<val_type> buffer;
    for (std::size_t s=0; s<Species::n_species; ++s) {
      
      pa[s].q   = in.q[s];
      pa[s].m   = in.m[s];
      pa[s].qdm = in.q[s]/in.m[s];

      for (std::size_t ix=0; ix<Species::Particle::dim_x; ++ix) {
        buffer.resize(pa[s].x[ix].size());
        for (int i=0;i<pa[s].x[ix].size();++i)
          buffer[i] =  udist(gen);

        thrust::copy(buffer.begin(),buffer.end(),pa[s].x[ix].begin());
      }
      for (std::size_t iv=0; iv<Species::Particle::dim_v; ++iv) {
        buffer.resize(pa[s].x[iv].size());
        for (int i=0;i<pa[s].x[iv].size();++i)
          buffer[i] =  gdist(gen);

        thrust::copy(buffer.begin(),buffer.end(),pa[s].v[iv].begin());
      }
    }
   
  }

};
