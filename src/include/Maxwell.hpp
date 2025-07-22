#pragma once 


#include <chrono>
#include <thrust/for_each.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/zip_iterator.h>

#define TGET(x,i) thrust::get<i>(x)

template <typename Container>
void periodic_boundary(Container& arr,int nl, int nr) {

  thrust::copy(arr.end()-nr-nl-1, arr.end()-nr-1, arr.begin());
  thrust::copy(arr.begin()+nl, arr.begin()+nl+nr+1, arr.end()-nr-1);

}

template <typename Container>
void setInitField(Container& arr, int n) {

  thrust::transform(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(n),
                    arr.begin(), [] __host__ __device__ (int i) {
                      return static_cast<float>(std::exp(-(i-3000)*(i-3000)/30000));
                    });

}

template <typename PIC>
struct Maxwell {

  using Cell = PIC::Cell;
  using Const = PIC::Constant;
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
  thrust::device_vector<val_type> send_buffer, recv_buffer;

  template <typename MPI>
  Maxwell(Cell &cell,MPI &mpi) : cus(mpi.s), comm(mpi.comm),flag_mpi(mpi.flag), 
    n_mpi(mpi.world_size), r_mpi(mpi.world_rank), n_comm(1+cell.nl[0]) { 
    recv_buffer.resize(n_comm);
    send_buffer.resize(n_comm);
  }
  void operator()(Cell &cell) {
    using namespace thrust::placeholders;
    for (int s=0; s<n_species; s++) {
      thrust::transform(cell.edens[s].begin()+cell.nl[0],
                        cell.edens[s].begin()+cell.nl[0]+1,
                        cell.edens[s].end()-1-cell.nr[0],
                        cell.edens[s].end()-1-cell.nr[0],  _1+_2);
      thrust::copy(cell.edens[s].end()-1-cell.nr[0],
                   cell.edens[s].end()-cell.nr[0],
                   cell.edens[s].begin()+cell.nl[0]);
      thrust::copy(cell.edens[s].end()-1-cell.nr[0]-cell.nl[0],
                   cell.edens[s].end()-cell.nr[0]-1,
                   cell.edens[s].begin());
      thrust::copy(cell.edens[s].begin()+cell.nl[0]+1,
                   cell.edens[s].begin()+cell.nl[0]+cell.nr[0]+1,
                   cell.edens[s].end()-cell.nr[0]);
    }

    auto start = std::chrono::high_resolution_clock::now(); 
    for (int d=0; d<dim_v; d++) {
      thrust::fill(cell.jfield[d].begin(),cell.jfield[d].end(),0);
      for (int s=0; s<n_species; s++) {
        
        thrust::transform(cell.jsfield[s][d].begin(),cell.jsfield[s][d].end(),
                          cell.jfield[d].begin(),    cell.jfield[d].begin(),
                          thrust::placeholders::_1+thrust::placeholders::_2);
      
      } // for species
      thrust::transform(cell.jfield[d].begin()+cell.nl[0],
                        cell.jfield[d].begin()+cell.nl[0]+1,
                        cell.jfield[d].end()-1-cell.nr[0],
                        cell.jfield[d].end()-1-cell.nr[0],  _1+_2);
      thrust::copy(cell.jfield[d].end()-1-cell.nr[0],
                   cell.jfield[d].end()-cell.nr[0],
                   cell.jfield[d].begin()+cell.nl[0]);
      thrust::copy(cell.jfield[d].end()-1-cell.nr[0]-cell.nl[0],
                   cell.jfield[d].end()-cell.nr[0]-1,
                   cell.jfield[d].begin());
      thrust::copy(cell.jfield[d].begin()+cell.nl[0]+1,
                   cell.jfield[d].begin()+cell.nl[0]+cell.nr[0]+1,
                   cell.jfield[d].end()-cell.nr[0]);

      ncclAllReduce(thrust::raw_pointer_cast(cell.jfield[d].data()),
                    thrust::raw_pointer_cast(cell.jfield[d].data()),
                    cell.ng1,ncclType.name,ncclSum,comm,cus); 

    } // for dim_v
    cudaStreamSynchronize(cus);

    val_type Cj = cell.dt*2.0*M_PI, C0 = Const::c*cell.dt/cell.h[0], 
             nr = cell.nr[0], nl = cell.nl[0];

    auto end = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //std::cout << "ncclAllReduce costs " << duration.count() << "us" << std::endl;
    
    for (int d=1; d<dim_v; d++) {
      thrust::adjacent_difference(cell.jfield[d].begin(),cell.jfield[d].end(),
                                  cell.jfield[d].begin(),thrust::plus<val_type>());
      thrust::for_each(cell.jfield[d].begin(),cell.jfield[d].end(),
                       []__host__ __device__ (val_type& x){ x *=.5; });
      thrust::copy(cell.jfield[d].end()-nr-1, cell.jfield[d].end(), cell.jfield[d].begin());
    }
    
    
    using namespace thrust::placeholders;
    thrust::transform(cell.efield[0].begin(),cell.efield[0].end(),
                      cell.jfield[0].begin(),cell.efield[0].begin(), 
                      _1 - Cj*_2); // cal Ex

    auto zit_calEy_begin = thrust::make_zip_iterator(cell.efield[1].begin()+nl,
                                                     cell.jfield[1].begin()+nl,
                                                     cell.bfield[2].begin()+1+nl,
                                                     cell.bfield[2].begin()+nl-1);
    auto zit_calEy_end   = thrust::make_zip_iterator(cell.efield[1].end()-nr,
                                                     cell.jfield[1].end()-nr,
                                                     cell.bfield[2].end()-nr+1,
                                                     cell.bfield[2].end()-1+nr);
    periodic_boundary(cell.jfield[1],nl,nr);
    periodic_boundary(cell.bfield[2],nl,nr);
    thrust::for_each(zit_calEy_begin, zit_calEy_end,
                     [Cj,C0]__host__ __device__ (Tuple4Ref t) {
                        TGET(t,0) += -C0*(TGET(t,2)-TGET(t,3)) - Cj*TGET(t,1);
                     });
    periodic_boundary(cell.efield[1],nl,nr);

    auto zit_calEz_begin = thrust::make_zip_iterator(cell.efield[2].begin()+nl,
                                                     cell.jfield[2].begin()+nl,
                                                     cell.bfield[1].begin()+nl+1,
                                                     cell.bfield[1].begin()+nl-1);
    auto zit_calEz_end   = thrust::make_zip_iterator(cell.efield[2].end()-nr,
                                                     cell.jfield[2].end()-nr,
                                                     cell.bfield[1].end()-nr+1,
                                                     cell.bfield[1].end()-1+nr);

    periodic_boundary(cell.jfield[2],nl,nr);
    periodic_boundary(cell.bfield[1],nl,nr);
    thrust::for_each(zit_calEz_begin, zit_calEz_end,
                     [Cj,C0]__host__ __device__ (Tuple4Ref t) {
                        TGET(t,0) += C0*(TGET(t,2)-TGET(t,3)) - Cj*TGET(t,1);
                     });
    periodic_boundary(cell.efield[2],nl,nr);

    auto zit_calBy_begin = thrust::make_zip_iterator(cell.bfield[1].begin()+nl,
                                                     cell.efield[2].begin()+nl+1,
                                                     cell.efield[2].begin()+nl-1);
    auto zit_calBy_end   = thrust::make_zip_iterator(cell.bfield[1].end()-nr,
                                                     cell.efield[2].end()-nr+1,
                                                     cell.efield[2].end()-nr-1);
    periodic_boundary(cell.efield[2],nl,nr);
    thrust::for_each(zit_calBy_begin, zit_calBy_end,
                     [C0]__host__ __device__ (Tuple3Ref t) {
                        TGET(t,0) += C0*(TGET(t,1)-TGET(t,2));
                     });
    periodic_boundary(cell.bfield[1],nl,nr);


    auto zit_calBz_begin = thrust::make_zip_iterator(cell.bfield[2].begin()+nl,
                                                     cell.efield[1].begin()+nl+1,
                                                     cell.efield[1].begin()+nl-1);
    auto zit_calBz_end   = thrust::make_zip_iterator(cell.bfield[2].end()-nr,
                                                     cell.efield[1].end()-nr+1,
                                                     cell.efield[1].end()-nr-1);
    periodic_boundary(cell.efield[1],nl,nr);
    thrust::for_each(zit_calBz_begin, zit_calBz_end,
                     [C0]__host__ __device__ (Tuple3Ref t) {
                        TGET(t,0) += -C0*(TGET(t,1)-TGET(t,2));
                     });
    periodic_boundary(cell.bfield[2],nl,nr);


  }
  
};


