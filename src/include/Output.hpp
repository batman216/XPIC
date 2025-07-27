
#include <fstream>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

template <typename PIC>
struct Output {

  using val_type = PIC::value_type;

  const int mpi_rank;

  std::size_t intv1,intv2;

  uint32_t separator = 0xFFFFFFFF;


  std::ofstream ofs_ef, ofs_bf, ofs_jf; 
  std::size_t eout_intv, bout_intv, jout_intv, px, pv;

  thrust::host_vector<val_type> cell_buf, particle_buf;

  template <typename Input, typename MPI,
            typename Species, typename Cell>
  Output(Input& in,MPI& mpi,
         Species& p, Cell& c) 
  : mpi_rank(mpi.world_rank),
    eout_intv(in.out_ef), bout_intv(in.out_bf), jout_intv(in.out_jf),
    intv1(in.out_interval_1), intv2(in.out_interval_2) {

    cell_buf.resize(c.ng1);

    if (eout_intv) ofs_ef.open("efield.out",std::ios::binary|std::ios::out);
    if (bout_intv) ofs_bf.open("bfield.out",std::ios::binary|std::ios::out);
    if (jout_intv) ofs_jf.open("jfield.out",std::ios::binary|std::ios::out);

  }

  template<typename Particle, typename Cell>
  void operator()(int step, Particle& p, Cell& c) {

    if (eout_intv && step%eout_intv==0 && mpi_rank==0)  
      for (int d=0; d<PIC::dim_v; ++d) {
        thrust::copy(c.efield[d].begin(),c.efield[d].end(),cell_buf.begin());
        ofs_ef.write(reinterpret_cast<const char*>(cell_buf.data()),
                     cell_buf.size()*sizeof(val_type));
        ofs_ef.write(reinterpret_cast<const char*>(&separator), sizeof(separator)); 
      }   
    if (bout_intv && step%bout_intv==0 && mpi_rank==0)  
      for (int d=0; d<PIC::dim_v; ++d) {
        thrust::copy(c.bfield[d].begin(),c.bfield[d].end(),cell_buf.begin());
        ofs_bf.write(reinterpret_cast<const char*>(cell_buf.data()),
                     cell_buf.size()*sizeof(val_type));
        ofs_bf.write(reinterpret_cast<const char*>(&separator), sizeof(separator)); 
      }   
    if (jout_intv && step%jout_intv==0 && mpi_rank==0)  
      for (int d=0; d<PIC::dim_v; ++d) {
        thrust::copy(c.jfield[d].begin(),c.jfield[d].end(),cell_buf.begin());
        ofs_jf.write(reinterpret_cast<const char*>(cell_buf.data()),
                     cell_buf.size()*sizeof(val_type));
        ofs_jf.write(reinterpret_cast<const char*>(&separator), sizeof(separator)); 
      }   

  }



  ~Output() { ofs_ef.close(); ofs_bf.close(); ofs_jf.close(); }
  

};
