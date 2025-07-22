
template<typename PIC>
using ValPtr = typename PIC::value_type* __restrict__;
template<typename PIC>
using IdxPtr = typename PIC::index_type* __restrict__;


template <typename PIC> __global__ 
void kernel_Particle2Cell(const typename PIC::index_type* __restrict__ row_idx,  
                          const typename PIC::value_type* __restrict__ val_nz,
                          const typename PIC::value_type* __restrict__ particle, 
                          typename PIC::value_type* cell, 
                          const typename PIC::index_type n_particle,
                          const typename PIC::index_type n_cell) {
  using idx_type   = PIC::index_type;
  using val_type   = PIC::value_type;
  static constexpr idx_type n_node = PIC::n_node;
  idx_type col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= n_particle) return;

  val_type x_val = particle[col];

  #pragma unroll
  for (int i = 0; i < n_node; ++i) {
    idx_type row = row_idx[i*n_particle+col];
    val_type val = val_nz[i*n_particle+col];
    atomicAdd(&cell[row], val * x_val);
  }
}

template <typename PIC>
struct Particle2Cell {

  using idx_type   = PIC::index_type;
  using val_type   = PIC::value_type;

  const idx_type np, ng;

  Particle2Cell(idx_type np, idx_type ng) : np(np),ng(ng) {}

  void operator()(const typename PIC::index_type* __restrict__ col_idx,  
                  const typename PIC::value_type* __restrict__ val_nz,
                  const typename PIC::value_type* __restrict__ particle, 
                  typename PIC::value_type* cell) {

    kernel_Particle2Cell<PIC><<<(np+255)/256,256, sizeof(val_type)*ng>>>
      (col_idx, val_nz, particle, cell, np,ng);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
      std::cerr << "CUDA Error after kernel: " 
                << cudaGetErrorString(err) << std::endl;

  }

};


template <typename PIC> __global__
void kernel_Cell2Particle(const IdxPtr<PIC> col_idx,  
                          const ValPtr<PIC> val_nz,
                          const ValPtr<PIC> cell, 
                          typename PIC::value_type* particle, 
                          typename PIC::index_type M) {
  using idx_type = typename PIC::index_type;
  static constexpr idx_type n_node = PIC::n_node;

  idx_type i = blockIdx.x*blockDim.x + threadIdx.x; 

  if (i<M) {
    particle[i] = 0;
    #pragma unroll
    for (idx_type node=0; node<n_node; ++node) {
      idx_type ii = i + M*node ;
      particle[i] += val_nz[ii] * cell[col_idx[ii]]; 
    }
  }

}

template <typename PIC>
struct Cell2Particle {

  using idx_type   = PIC::index_type;
  using val_type   = PIC::value_type;

  idx_type np;

  Cell2Particle(idx_type np) : np(np) {}

  void operator()(const IdxPtr<PIC> col_idx, 
                  const ValPtr<PIC> val_nz,
                  const ValPtr<PIC> cell,
                  typename PIC::value_type* particle) {

    kernel_Cell2Particle<PIC><<<(np+255)/256,256>>>(col_idx, val_nz,
                                                    cell, particle, np);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
      std::cerr << "CUDA Error after kernel: " 
                << cudaGetErrorString(err) << std::endl;

  }

};
