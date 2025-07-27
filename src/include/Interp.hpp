
#include <chrono>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <nccl.h>
#include "MPIWrapper.hpp"
#include "XPIC_HelperFunctions.hpp"
#include "detail/interpolation.inl"
#include "detail/xpic_traits.hpp"
#include "detail/index_weight.hpp"

#define RPC(x) thrust::raw_pointer_cast(x.data())
#define TGET(x,i) thrust::get<i>(x)
template <typename Func,typename...Arg>
__host__ __device__
auto apply(Func func, Arg... args) {
  return func(args...);
}

template <typename PIC>
struct calVal {
  using idx_type   = PIC::index_type;
  using val_type   = PIC::value_type;
  static constexpr idx_type n_node = PIC::n_node;
  static constexpr idx_type order  = PIC::interp_order;
  static constexpr idx_type xdim  = PIC::dim_x;

  using ParticleTupleRef = typename tupleTraits<val_type,idx_type,xdim>::ParticleTupleRef;
  using IntpTupleRef     = typename tupleTraits<val_type,idx_type,xdim>::IntpTupleRef;
  using WeightTupleRef   = typename tupleTraits<val_type,idx_type,xdim>::WeightTupleRef;

  std::array<idx_type,xdim> ng,nl;

  __host__ __device__
  calVal(const std::array<idx_type,xdim>& ng,
         const std::array<idx_type,xdim>& nl)
  : ng(ng),nl(nl) {}


  template <idx_type...idx>
  __host__ __device__
  auto calIndex(std::index_sequence<idx...>,
                std::array<val_type,xdim> index,
                std::array<idx_type,xdim> ng) {
    // 计算每个节点(1d->2/2d->4/3d->8)在一维内存中的索引(即插值矩阵的非0行标)
    return thrust::make_tuple(::apply(index_of_each_node<order,idx>(),index,ng)...);
  }

  template <idx_type...idx>
  __host__ __device__
  auto calWeight(std::index_sequence<idx...>,
                 std::array<val_type,xdim> wei) {
    return thrust::make_tuple(::apply(weight_to_each_node<order,idx>(),wei)...);
  }

  __host__ __device__
  void operator()(IntpTupleRef t)  {

    std::array<val_type,xdim> idx,wei;

    // 注意ghost网格带来的偏移 nl
    wei[0] = std::modf((TGET(TGET(t,0/*particle*/),0)+nl[0]),idx.data());
    if constexpr (xdim>1)
      wei[1] = std::modf(TGET(TGET(t,0),1),idx.data()+1);
    if constexpr (xdim>2)
      wei[2] = std::modf(TGET(TGET(t,0),2),idx.data()+2);

    TGET(t,1/*index*/)  = calIndex (std::make_index_sequence<n_node>{},idx,ng);
    TGET(t,2/*weight*/) = calWeight(std::make_index_sequence<n_node>{},wei);

  }

};

template<typename PIC>
struct SpMV {
  using idx_type   = PIC::index_type;
  using val_type   = PIC::value_type;
  using wei_type   = PIC::weight_type;

  cudaDataType_t cudaDataType
    = cudaDataTypeTraits<val_type>::type();
  cusparseIndexType_t cusparseIndexType
    = cusparseIndexTypeTraits<idx_type>::type();

  cusparseSpMatDescr_t matA = nullptr;
  cusparseDnVecDescr_t X = nullptr, Y = nullptr;
  cusparseHandle_t     handle     = nullptr;
  void*                dBuffer    = nullptr;
  size_t               bufferSize = 0;
  const val_type       alpha = 1., beta = 0.;

  cusparseOperation_t  cuop;
  bool                 transpose;

  void init(idx_type* A_ccsr, idx_type* A_cols, wei_type* A_vals,
            idx_type  n_rows, idx_type  n_cols, idx_type  n_nz,
            bool transpose,   wei_type* dataX,  val_type* dataY) {

    cusparseCreate(&handle);
    cusparseCreateCsr(&matA,
                      n_rows, n_cols, n_nz, // cols and rows inversed (transpose)
                      A_ccsr, A_cols, A_vals,
                      cusparseIndexType, cusparseIndexType,
                      CUSPARSE_INDEX_BASE_ZERO, cudaDataType);

    int nx = transpose? n_rows : n_cols;
    int ny = transpose? n_cols : n_rows;
    cuop   = transpose? CUSPARSE_OPERATION_TRANSPOSE
                      : CUSPARSE_OPERATION_NON_TRANSPOSE;

    cusparseCreateDnVec(&X, nx, dataX, cudaDataType);
    cusparseCreateDnVec(&Y, ny, dataY, cudaDataType);
    cusparseSpMV_bufferSize(handle,cuop,
                            &alpha,matA,X,
                            &beta,      Y, cudaDataType,
                            CUSPARSE_SPMV_CSR_ALG2,&bufferSize);

    cudaMalloc(&dBuffer,bufferSize);
  }
  void run() {
    cusparseSpMV(handle, cuop, &alpha, matA, X, &beta, Y,
                 cudaDataType, CUSPARSE_SPMV_CSR_ALG2, dBuffer);
  }

  void run(wei_type* dataX,  val_type* dataY) {
    cusparseDnVecSetValues(X,dataX);
    cusparseDnVecSetValues(Y,dataY);
    cusparseSpMV(handle, cuop, &alpha, matA, X, &beta, Y,
                 cudaDataType, CUSPARSE_SPMV_CSR_ALG2, dBuffer);
  }

};



template<typename PIC>
struct Interp {

  using Container  = typename PIC::ParticleContainer;
  using idx_type   = PIC::index_type;
  using val_type   = PIC::value_type;
  using wei_type   = PIC::weight_type;
  using Species    = PIC::Species;
  using Cell       = PIC::Cell;
  static constexpr int n_species = static_cast<idx_type>(PIC::n_species),
                           dim_x = static_cast<idx_type>(PIC::dim_x),
                           dim_v = static_cast<idx_type>(PIC::dim_v);

  static constexpr idx_type n_node = PIC::n_node;

  nccl_traits<val_type> ncclType;
  cudaStream_t cus;
  ncclComm_t comm;

  std::array<thrust::device_vector<idx_type>,n_species> A_cols;
  std::array<thrust::device_vector<val_type>,n_species> A_vals, ones;

  using ParticleZipItor = tupleTraits<val_type,idx_type,dim_x>::ParticleZipItor;
  using IndexZipItor    = tupleTraits<val_type,idx_type,dim_x>::IndexZipItor;
  using WeightZipItor   = tupleTraits<val_type,idx_type,dim_x>::WeightZipItor;
  using IntpZipItor     = thrust::zip_iterator<
                          thrust::tuple<ParticleZipItor,IndexZipItor,WeightZipItor>>;

  ParticleZipItor z_itor_par;
  IndexZipItor    z_itor_col;
  WeightZipItor   z_itor_val;
  IntpZipItor     z_itor_pcv;

  idx_type n_rows[n_species], n_cols, n_nz[n_species];

  template <typename MPI>
  Interp(Cell& cell,MPI &mpi) : n_cols(cell.ng1), cus(mpi.s), comm(mpi.comm) {
    n_cols = cell.ng1;
  }

  void calWeight(Species& sp, Cell& cell) {

    for (int s=0; s<n_species; ++s) {
      n_rows[s] = sp[s].np;
      n_nz[s]   = n_rows[s]*n_node; // #none-zero elements

      // prepare to assemble the sparse matrix (i,j,val)
      A_cols[s].resize(n_nz[s]);  // column index
      A_vals[s].resize(n_nz[s]);  // element value

      // 23dimension tomultiply
      z_itor_par = thrust::make_zip_iterator(sp[s].x[0].begin());
      z_itor_col = thrust::make_zip_iterator(A_cols[s].begin()+0*n_rows[s],
                                             A_cols[s].begin()+1*n_rows[s],
                                             A_cols[s].begin()+2*n_rows[s],
                                             A_cols[s].begin()+3*n_rows[s]);
      z_itor_val = thrust::make_zip_iterator(A_vals[s].begin()+0*n_rows[s],
                                             A_vals[s].begin()+1*n_rows[s],
                                             A_vals[s].begin()+2*n_rows[s],
                                             A_vals[s].begin()+3*n_rows[s]);
      z_itor_pcv = thrust::make_zip_iterator(z_itor_par,z_itor_col,z_itor_val);

      // from particle position to calculate the index: A_cols,
      // and the cooresponding weight: A_vals
      thrust::for_each(z_itor_pcv,z_itor_pcv+n_rows[s],
                       calVal<PIC>(cell.ng_al,cell.nl));

      val_type qw = sp[s].q *sp[s].w;
      ones[s].resize(n_rows[s]);
      thrust::fill(ones[s].begin(),ones[s].end(),qw);

      thrust::fill(cell.edens[s].begin(),cell.edens[s].end(),0);
      for (int d=0; d<dim_v; d++)
        thrust::fill(cell.jsfield[s][d].begin(),cell.jsfield[s][d].end(),0);

      /*
      cuSpMV[s].init(thrust::raw_pointer_cast(A_ccsr[s].data()),
                     thrust::raw_pointer_cast(A_cols[s].data()),
                     thrust::raw_pointer_cast(A_vals[s].data()),
                     n_rows[s], n_cols, n_nz[s], true,
                     thrust::raw_pointer_cast(ones[s].data()),
                     thrust::raw_pointer_cast(cell.edens[s].data()));
                     */
    }// for species
  }
  void particle2cell(Species& sp, Cell& cell) {
    using namespace thrust::placeholders;

    val_type nr = cell.nr[0], nl = cell.nl[0],
             CJ = cell.h[0]/cell.dt;

    thrust::copy(cell.eden.begin(),cell.eden.end(),cell.eden_buf.begin()); 
    thrust::fill(cell.eden.begin(),cell.eden.end(),0); 
    for (int d=0; d<dim_v; d++) 
      thrust::fill(cell.jfield[d].begin(),cell.jfield[d].end(),0);
    // 暂时先这样，每次都把电流插值好，算VB电流再这个覆盖掉就行了
    
    // charge density sum
    for (int s=0; s<n_species; ++s) {

      Particle2Cell<PIC> p2c(n_rows[s],n_cols);

      val_type qw = sp[s].q *sp[s].w;
      p2c(RPC(A_cols[s]), RPC(A_vals[s]),
          RPC(ones[s]),   RPC(cell.edens[s]));

//      periodicBoundaryInterpSum(cell.edens[s],nl,nr);

      thrust::transform(cell.edens[s].begin(),cell.edens[s].end(),
                        cell.eden.begin(),cell.eden.begin(),_1+_2);

    } // for species
    ncclAllReduce(thrust::raw_pointer_cast(cell.edens[1].data()),
                  thrust::raw_pointer_cast(cell.edens[1].data()),
                    cell.ng1,ncclType.name,ncclSum,comm,cus);

    ncclAllReduce(thrust::raw_pointer_cast(cell.eden.data()),
                  thrust::raw_pointer_cast(cell.eden.data()),
                    cell.ng1,ncclType.name,ncclSum,comm,cus);

    periodicBoundary(cell.eden,nl,nr);
    // current sum
    for (int d=0; d<dim_v; d++) {
      for (int s=0; s<n_species; s++) {
        Particle2Cell<PIC> p2c(n_rows[s],n_cols);
        p2c(RPC(A_cols[s]),  RPC(A_vals[s]),
            RPC(sp[s].v[d]), RPC(cell.jsfield[s][d]));

        // 乘上电荷以及粒子权重
        val_type qw = sp[s].q *sp[s].w;
        thrust::for_each(cell.jsfield[s][d].begin(),cell.jsfield[s][d].end(),
                         [qw]__host__ __device__(val_type& val) { val *= qw; });
        periodicBoundaryInterpSum(cell.jsfield[s][d],nl,nr);
        thrust::transform(cell.jsfield[s][d].begin(),cell.jsfield[s][d].end(),
                          cell.jfield[d].begin(),    cell.jfield[d].begin(),
                          thrust::placeholders::_1+thrust::placeholders::_2);
      } // for species

      ncclAllReduce(thrust::raw_pointer_cast(cell.jfield[d].data()),
                    thrust::raw_pointer_cast(cell.jfield[d].data()),
                    cell.ng1,ncclType.name,ncclSum,comm,cus); 
      periodicBoundary(cell.jfield[d],nl,nr);
    } // for dim_v

  }


  void cell2particle(Species& sp, Cell& cell) {

    thrust::device_vector<val_type> buf(cell.ng1+1);
    for (int s=0; s<n_species; ++s) {

      Cell2Particle<PIC> c2p(n_rows[s]);
      c2p(RPC(A_cols[s]), RPC(A_vals[s]), RPC(cell.bfield[0]), RPC(sp[s].B[0]));
      c2p(RPC(A_cols[s]), RPC(A_vals[s]), RPC(cell.efield[1]), RPC(sp[s].E[1]));
      c2p(RPC(A_cols[s]), RPC(A_vals[s]), RPC(cell.efield[2]), RPC(sp[s].E[2]));
      c2p(RPC(A_cols[s]), RPC(A_vals[s]), RPC(cell.efield[0]), RPC(sp[s].E[0]));
      c2p(RPC(A_cols[s]), RPC(A_vals[s]), RPC(cell.bfield[1]), RPC(sp[s].B[1]));
      c2p(RPC(A_cols[s]), RPC(A_vals[s]), RPC(cell.bfield[2]), RPC(sp[s].B[2]));

    } // for species

  }
};
