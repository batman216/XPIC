#pragma once
#include <thrust/partition.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include "MPIWrapper.hpp"



template <typename Array, std::size_t...idx>
auto make_zip_iterator_from_array(Array &arr,std::index_sequence<idx...>) {
  return thrust::make_zip_iterator(arr[idx].begin()...); 
}

template <typename PIC>
struct Swap {

  using val_type  = PIC::value_type;
  static constexpr int xdim  = PIC::dim_x;
  static constexpr int vdim  = PIC::dim_v;
  static constexpr int xvdim = PIC::dim_x+PIC::dim_v;

  nccl_traits<val_type> ncclType;
  ncclComm_t comm;
  cudaStream_t cus;

  int n_mpi,r_mpi;
  char flag_mpi;

  val_type A,B,C;
  std::array<val_type,2> bound;
  std::array<thrust::device_vector<val_type>,xvdim> send_buffer;
  std::array<thrust::device_vector<val_type>,xvdim> r_recv_buffer, l_recv_buffer;

  template <typename Cell, typename MPI>
  Swap(Cell &cell, MPI &mpi) : cus(mpi.s), comm(mpi.comm),flag_mpi(mpi.flag), 
  n_mpi(mpi.world_size), r_mpi(mpi.world_rank) {

    bound[1] = cell.b[0];
    bound[0] = cell.a[0];

    A = cell.a_gl[0];
    B = cell.b_gl[0];
    C = cell.L_gl[0];
  }


  template <typename Species>
  void operator()(Species& sp) {
    val_type a = bound[0], b = bound[1];
    for (int s=0; s<PIC::n_species; ++s) {
      std::size_t np = sp[s].np;
      // todo
      auto zit = thrust::make_zip_iterator(sp[s].x[0].begin(),sp[s].v[0].begin(),
                                           sp[s].v[1].begin(),sp[s].v[2].begin());
      using Tuple = thrust::tuple<val_type,val_type,val_type,val_type>;

      // 把超出边界[a,b)的粒子都放到数组最后
      auto mid = thrust::stable_partition(zit,zit+np,[a,b]
                          __host__ __device__(Tuple t)
                          { return thrust::get<0>(t)>=a&&thrust::get<0>(t)<b;});
      // 这一步完成后，[a,b)之间的粒子会放在0到mid, 超出范围的放在mid之后

      // 需要保留的粒子数和需要发送走的粒子数
      std::size_t n_remain = static_cast<std::size_t>(mid-zit),
                  n_send   = np - n_remain;

      // 在需要发送走的粒子中，往左发送的放前面
      auto midlr = thrust::stable_partition(mid,mid+n_send,[a]__host__ __device__(Tuple t)
                           { return thrust::get<0>(t)<a;});

      // 往左发送的粒子数和往右发送的粒子数
      std::size_t n_l_send = static_cast<std::size_t>(midlr-mid),
                  n_r_send = n_send - n_l_send;
   
      for (int d=0; d<xvdim; d++) send_buffer[d].resize(n_send);

      auto zit_send_buf = make_zip_iterator_from_array(send_buffer,
                                                  std::make_index_sequence<xvdim>{});
      thrust::copy(mid,mid+n_send,zit_send_buf);
   
      int l_rank = r_mpi==0? n_mpi - 1  : r_mpi-1,
          r_rank = r_mpi==n_mpi - 1 ? 0 : r_mpi+1;
      
      int n_r_recv{}, n_l_recv{};

      // 告诉左边的核我要给你多少粒子
      MPI_Send(&n_l_send,1,MPI_INT,l_rank,0,MPI_COMM_WORLD);
      // 听右边的核说他要给我多少粒子
      MPI_Recv(&n_r_recv,1,MPI_INT,r_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      // 告诉右边的核我要给你多少粒子
      MPI_Send(&n_r_send,1,MPI_INT,r_rank,0,MPI_COMM_WORLD);
      // 听左边的核说他要给我多少粒子
      MPI_Recv(&n_l_recv,1,MPI_INT,l_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      ncclGroupStart();      
      for (int d=0; d<xvdim; d++) {
        // 开辟好左右缓冲区准备接收粒子
        r_recv_buffer[d].resize(n_r_recv);
        l_recv_buffer[d].resize(n_l_recv);

        ncclResult_t nccl_; 

        // <-----向左发送--------
        //if (r_mpi>0)
        nccl_ = ncclSend(thrust::raw_pointer_cast(send_buffer[d].data()),
                         n_l_send,ncclType.name,l_rank,comm,cus);
        if (nccl_!=ncclSuccess) 
          std::cerr << "NCCL Send failed: " << ncclGetErrorString(nccl_) << std::endl;
        //if (r_mpi<n_mpi-1)
        nccl_ = ncclRecv(thrust::raw_pointer_cast(r_recv_buffer[d].data()),
                         n_r_recv,ncclType.name,r_rank,comm,cus);
        if (nccl_!=ncclSuccess) 
          std::cerr << "NCCL Recv failed: " << ncclGetErrorString(nccl_) << std::endl;
      
        // -------向右发送-------->
        //if (r_mpi<n_mpi-1)
        nccl_ = ncclSend(thrust::raw_pointer_cast(send_buffer[d].data()+n_l_send),
                         n_r_send,ncclType.name,r_rank,comm,cus);
        if (nccl_!=ncclSuccess) 
          std::cerr << "NCCL Send failed: " << ncclGetErrorString(nccl_) << std::endl;
        //if (r_mpi>0)
        nccl_ = ncclRecv(thrust::raw_pointer_cast(l_recv_buffer[d].data()),
                         n_l_recv,ncclType.name,l_rank,comm,cus);
        if (nccl_!=ncclSuccess) 
          std::cerr << "NCCL Recv failed: " << ncclGetErrorString(nccl_) << std::endl;
      }
      ncclGroupEnd();
      np += - n_send + n_r_recv + n_l_recv;

      for (int d=0; d<xdim; d++) 
        sp[s].x[d].resize(np);
      for (int d=0; d<vdim; d++) { 
        sp[s].v[d].resize(np);
        sp[s].E[d].resize(np);
        sp[s].B[d].resize(np);
      }

      sp[s].np = np; 

      /// resize了之后，zip_iterator 需要重新初始化
      // todo
      zit = thrust::make_zip_iterator(sp[s].x[0].begin(),sp[s].v[0].begin(),
                                      sp[s].v[1].begin(),sp[s].v[2].begin());
      auto zit_rrecvbuf = make_zip_iterator_from_array(r_recv_buffer,
                                                       std::make_index_sequence<xvdim>{});
      auto zit_lrecvbuf = make_zip_iterator_from_array(l_recv_buffer,
                                                       std::make_index_sequence<xvdim>{});
      //if (r_mpi>0)
        thrust::copy(zit_lrecvbuf,zit_lrecvbuf+n_l_recv,zit+n_remain);
      //if (r_mpi<n_mpi-1)
        thrust::copy(zit_rrecvbuf,zit_rrecvbuf+n_r_recv,zit+n_remain+n_l_recv);
     
      val_type L_gl = C, a_gl = A, b_gl = B;
      thrust::transform_if(sp[s].x[0].begin(),sp[s].x[0].end(),sp[s].x[0].begin(),
                           [L_gl]__host__ __device__(val_type& val) 
                           { return val + L_gl; },
                           [a_gl]__host__ __device__(val_type& val) 
                           { return val < a_gl; });
      thrust::transform_if(sp[s].x[0].begin(),sp[s].x[0].end(),sp[s].x[0].begin(),
                           [L_gl]__host__ __device__(val_type& val) 
                           { return val - L_gl; },
                           [b_gl]__host__ __device__(val_type& val) 
                           { return val > b_gl; });
      
      // todo: 此处不shuffle会影响后续的插值正确性，why?
      thrust::default_random_engine g;
      thrust::shuffle(zit,zit+np,g);
    }
  }
};




