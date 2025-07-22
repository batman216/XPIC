

#pragma once
#include <fstream>
#include <string>

namespace diag {

template <typename PIC>
struct TraceParticle {

  std::ofstream out;
  int mpi_rank;

  template <typename MPI>
  TraceParticle(MPI mpi) :mpi_rank(mpi.world_rank) {
    if (mpi_rank==0)
      out.open("ParticleTrace.out",std::ios::out);
  }
  ~TraceParticle() { if(mpi_rank==0) out.close(); }

  void operator()(PIC::Particle& p) {

    for (int ix=0; ix<PIC::dim_x; ix++)
      out << p.x[ix][10] << "\t";
    for (int iv=0; iv<PIC::dim_v; iv++)
      out << p.v[iv][10]  << "\t";
    out << std::endl;
  }

};

template <typename PIC>
struct Energy {

  std::ofstream out;
  int mpi_rank;

  template <typename MPI>
  Energy(MPI mpi) :mpi_rank(mpi.world_rank) {
    if (mpi_rank==0)
      out.open("Energy",std::ios::out);
  }
  ~Energy() { if(mpi_rank==0) out.close(); }

  void operator()(PIC::Species& sp, PIC::Cell &cell) {

  }



};



}
