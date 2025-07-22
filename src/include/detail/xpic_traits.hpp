
template <typename val_type, typename idx_type, int xdim>
struct tupleTraits;

template <typename val_type, typename idx_type>
struct tupleTraits<val_type,idx_type,1> {

  using ValItor  = thrust::device_vector<val_type>::iterator;
  using WeiItor  = thrust::device_vector<val_type>::iterator;
  using IdxItor  = thrust::device_vector<idx_type>::iterator;

  using ParticleZipItor = thrust::zip_iterator<thrust::tuple<ValItor>>;
  using IndexZipItor    = thrust::zip_iterator<thrust::tuple<IdxItor,IdxItor,IdxItor,IdxItor>>;
  using WeightZipItor   = thrust::zip_iterator<thrust::tuple<WeiItor,WeiItor,WeiItor,WeiItor>>;
  using IntpZipItor     = thrust::zip_iterator<
                            thrust::tuple<ParticleZipItor,IndexZipItor,WeightZipItor>>;

  using ParticleTupleRef = thrust::detail::
                             tuple_of_iterator_references<val_type&>;
  using IndexTupleRef    = thrust::detail::
                             tuple_of_iterator_references<
                               idx_type&,idx_type&,idx_type&,idx_type&>;
  using WeightTupleRef   = thrust::detail::
                             tuple_of_iterator_references<
                               val_type&,val_type&,val_type&,val_type&>;
  using IntpTupleRef     = thrust::detail::
                             tuple_of_iterator_references
                               <ParticleTupleRef,IndexTupleRef,WeightTupleRef>;
};


template <typename val_type, typename idx_type>
struct tupleTraits<val_type,idx_type,2> {

  using ValItor  = thrust::device_vector<val_type>::iterator;
  using IdxItor  = thrust::device_vector<idx_type>::iterator;

  using ParticleZipItor = thrust::zip_iterator<thrust::tuple<ValItor,ValItor>>;
  using IndexZipItor    = thrust::zip_iterator<thrust::tuple<IdxItor,IdxItor,
                                                             IdxItor,IdxItor>>;
  using WeightZipItor   = thrust::zip_iterator<thrust::tuple<ValItor,ValItor,
                                                             ValItor,ValItor>>;
  using IntpZipItor     = thrust::zip_iterator<
                          thrust::tuple<ParticleZipItor,IndexZipItor,WeightZipItor>>;

  using ParticleTupleRef = thrust::detail::
                      tuple_of_iterator_references<
                      val_type&,val_type&>;
  using WeightTupleRef = thrust::detail::
                       tuple_of_iterator_references<
                       val_type&,val_type&,val_type&,val_type&>;


  using IndexTupleRef = thrust::detail::
                       tuple_of_iterator_references<
                       idx_type&,idx_type&,idx_type&,idx_type&>;
 
  using IntpTupleRef = thrust::detail::
                            tuple_of_iterator_references
                            <ParticleTupleRef,IndexTupleRef,WeightTupleRef>;
};

template <typename val_type, typename idx_type>
struct tupleTraits<val_type,idx_type,3> {

  using ValItor  = thrust::device_vector<val_type>::iterator;
  using IdxItor  = thrust::device_vector<idx_type>::iterator;

  using ParticleZipItor = thrust::zip_iterator<thrust::tuple<ValItor,ValItor,ValItor>>;
  using IndexZipItor    = thrust::zip_iterator<thrust::tuple<IdxItor,IdxItor,
                                                             IdxItor,IdxItor, 
                                                             IdxItor,IdxItor,
                                                             IdxItor,IdxItor>>;
  using WeightZipItor   = thrust::zip_iterator<thrust::tuple<ValItor,ValItor,
                                                             ValItor,ValItor,
                                                             ValItor,ValItor,
                                                             ValItor,ValItor>>;
  using IntpZipItor     = thrust::zip_iterator<
                          thrust::tuple<ParticleZipItor,IndexZipItor,WeightZipItor>>;

  using ParticleTupleRef = thrust::detail::
                             tuple_of_iterator_references<
                               val_type&,val_type&,val_type&>;
  using IndexTupleRef = thrust::detail::
                          tuple_of_iterator_references<
                            idx_type&,idx_type&,idx_type&,idx_type&,
                              idx_type&,idx_type&,idx_type&,idx_type&>;
  using WeightTupleRef = thrust::detail::
                           tuple_of_iterator_references<
                             val_type&,val_type&,val_type&,val_type&,
                               val_type&,val_type&,val_type&,val_type&>;
  using IntpTupleRef = thrust::detail::
                        tuple_of_iterator_references
                          <ParticleTupleRef,IndexTupleRef,WeightTupleRef>;
};


template <typename idx_type>
struct cusparseIndexTypeTraits {
  static constexpr cusparseIndexType_t type() {
    if constexpr (sizeof(idx_type)==4)
      return CUSPARSE_INDEX_32I;
    else if  (sizeof(idx_type)==8)
      return CUSPARSE_INDEX_64I;
  }
};


template <typename val_type>
struct cudaDataTypeTraits {
  static constexpr cudaDataType_t type() {
    if constexpr (sizeof(val_type)==4)
      return CUDA_R_32F;
    else if  (sizeof(val_type)==8)
      return CUDA_R_64F;
  }
};


