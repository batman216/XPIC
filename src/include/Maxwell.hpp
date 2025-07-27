    
#pragma once 

#include <chrono>
#include <thrust/for_each.h>
#include <thrust/adjacent_difference.h>
#include <thrust/scan.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>

#include "XPIC_HelperFunctions.hpp"

#define TGET(x,i) thrust::get<i>(x)

template <typename PIC>
void calVBCurrent(typename PIC::Cell& cell) {

  using namespace thrust::placeholders;
  using val_type = PIC::value_type;
  // Villasenor&Buneman current
  val_type CJ = 1.0/cell.dt;
  thrust::transform(cell.eden.begin(),cell.eden.end(),
                    cell.eden_buf.begin(), cell.eden_buf.begin(),CJ*(_2-_1)); 
  thrust::exclusive_scan(cell.eden_buf.begin(),cell.eden_buf.end(),cell.jfield[0].begin());
  val_type mean = thrust::reduce(cell.jfield[0].begin(),cell.jfield[0].end(),0)/static_cast<val_type>(cell.ng1);
  //std::cout << mean << std::endl;
  thrust::for_each(cell.jfield[0].begin(),cell.jfield[0].end(),
                    [mean]__device__(val_type& val) { val-=mean; });
}

template <typename PIC>
struct Maxwell {

  using Cell = PIC::Cell;
  using Const = PIC::Constant;
  using idx_type = PIC::index_type;
  using val_type = PIC::value_type;
  static constexpr std::size_t n_species = PIC::n_species, 
                                   dim_x = PIC::dim_x, 
                                   dim_v = PIC::dim_v;

  using Tuple3Ref = thrust::detail::
                      tuple_of_iterator_references<
                        val_type&,val_type&,val_type&>;
  using Tuple4Ref = thrust::detail::
                      tuple_of_iterator_references<
                        val_type&,val_type&,val_type&,val_type&>;
  using MaxwellTupleRef2 = thrust::detail::
                            tuple_of_iterator_references
                             <Tuple3Ref,Tuple3Ref>;
  using MaxwellTupleRef3 = thrust::detail::
                            tuple_of_iterator_references
                             <Tuple3Ref,Tuple3Ref,Tuple3Ref>;
  nccl_traits<val_type> ncclType;
  ncclComm_t comm;
  cudaStream_t cus;

  int n_mpi,r_mpi;
  char flag_mpi;
  
  std::size_t n_comm;
  thrust::device_vector<val_type> F1,F2,G1,G2;

  template <typename MPI>
  Maxwell(Cell &cell,MPI &mpi) : cus(mpi.s), comm(mpi.comm),flag_mpi(mpi.flag), 
    n_mpi(mpi.world_size), r_mpi(mpi.world_rank), n_comm(1+cell.nl[0]) { 

    F1.resize(cell.ng1);
    F2.resize(cell.ng1);
    G1.resize(cell.ng1);
    G2.resize(cell.ng1);
    idx_type nr = cell.nr[0], nl = cell.nl[0];
    periodicBoundaryHalf(cell.efield[0],nl,nr);    
    periodicBoundaryHalf(cell.efield[1],nl,nr);    
    periodicBoundaryHalf(cell.efield[2],nl,nr);    
    periodicBoundaryHalf(cell.bfield[0],nl,nr);    
    periodicBoundaryHalf(cell.bfield[1],nl,nr);    
    periodicBoundaryHalf(cell.bfield[2],nl,nr);    



    using namespace thrust::placeholders;
    thrust::transform(cell.efield[1].begin(),cell.efield[1].end(),
                      cell.bfield[2].begin(),F1.begin(), 0.5*(_1+_2)); 
    thrust::transform(cell.efield[1].begin(),cell.efield[1].end(),
                      cell.bfield[2].begin(),F2.begin(), 0.5*(_1-_2)); 
    thrust::transform(cell.efield[2].begin(),cell.efield[2].end(),
                      cell.bfield[1].begin(),G1.begin(), 0.5*(_1+_2)); 
    thrust::transform(cell.efield[2].begin(),cell.efield[2].end(),
                      cell.bfield[1].begin(),G2.begin(), 0.5*(_1-_2)); 
  }


  void operator()(Cell &cell) {
    using namespace thrust::placeholders;

    val_type Cj = cell.dt*2.0*M_PI;
    idx_type nr = cell.nr[0], nl = cell.nl[0], ncell = cell.n_cell[0];

    for (int d=0; d<dim_v; d++) {
      smooth(cell.jfield[d],nl,nr);
      smooth(cell.jfield[d],nl,nr);
    }
    periodicBoundaryHalf(cell.eden,nl,nr);    
    periodicBoundaryHalf(F1,nl,nr);
    periodicBoundaryHalf(F2,nl,nr);
    periodicBoundaryHalf(G1,nl,nr);
    periodicBoundaryHalf(G2,nl,nr);

    if constexpr (PIC::dim_x == 1) {


      // Transverse Field

      thrust::transform(F1.begin()+nl,F1.end()-nr,cell.jfield[1].begin()+nl,
                        F1.begin()+nl+1, _1-Cj*_2); F1[nl] = F1[nl+ncell];
      thrust::transform(F2.begin()+nl,F2.end()-nr,cell.jfield[1].begin()+nl,
                        F2.begin()+nl-1, _1-Cj*_2); F2[nl+ncell] = F2[nl];
      thrust::transform(G1.begin()+nl,G1.end()-nr,cell.jfield[2].begin()+nl,
                        G1.begin()+nl+1, _1-Cj*_2); G1[nl] = G1[nl+ncell];
      thrust::transform(G2.begin()+nl,G2.end()-nr,cell.jfield[2].begin()+nl,
                        G2.begin()+nl-1, _1-Cj*_2); G2[nl+ncell] = G2[nl];

      thrust::transform(F1.begin(),F1.end(),F2.begin(),
                        cell.efield[1].begin(), _1+_2);
      thrust::transform(G1.begin(),G1.end(),G2.begin(),
                        cell.efield[2].begin(), _1+_2);
      thrust::transform(F1.begin(),F1.end(),F2.begin(),
                        cell.bfield[2].begin(), _1-_2);
      thrust::transform(G1.begin(),G1.end(),G2.begin(),
                        cell.bfield[1].begin(), _1-_2);

      // Longitudinal Field

      thrust::transform(cell.efield[0].begin(),cell.efield[0].end(),cell.jfield[0].begin(),
                        cell.efield[0].begin(),_1-_2*2.0*Cj);

    }
    periodicBoundaryHalf(F1,nl,nr);
    periodicBoundaryHalf(F2,nl,nr);
    periodicBoundaryHalf(G1,nl,nr);
    periodicBoundaryHalf(G2,nl,nr);

    periodicBoundaryHalf(cell.efield[0],nl,nr);    
    periodicBoundaryHalf(cell.efield[1],nl,nr);    
    periodicBoundaryHalf(cell.efield[2],nl,nr);    
    periodicBoundaryHalf(cell.bfield[0],nl,nr);    
    periodicBoundaryHalf(cell.bfield[1],nl,nr);    
    periodicBoundaryHalf(cell.bfield[2],nl,nr);    




    periodicBoundaryHalf(cell.efield[0],nl,nr);
    periodicBoundaryHalf(cell.bfield[1],nl,nr);
    periodicBoundaryHalf(cell.bfield[2],nl,nr);

  }
  
};


