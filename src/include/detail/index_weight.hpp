// ----------------- 
// |       |       |
// |   3   |   2   |
// |       |       |
// |---------------|
// |       |       |
// |   0   |   1   |
// |       |       |
// -----------------


template <int order,int i>
struct index_of_each_node;

template <int order,int i>
struct weight_to_each_node;


template<>
struct index_of_each_node<1,0> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3)
      return std::min(static_cast<idx_type>(idx[0]+idx[1]*ng[0]+idx[2]*ng[1]*ng[0]),
                 ng[0]*ng[1]*ng[2]);
    else if constexpr (idx.size()==2)
      return std::min(static_cast<idx_type>(idx[0]+idx[1]*ng[0]), ng[0]*ng[1]);
    else if constexpr (idx.size()==1)
      return std::min(static_cast<idx_type>(idx[0]),ng[0]);
    else {}
  }
};

template<>
struct weight_to_each_node<1,0> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    if constexpr (wei.size()==3)
      return (1.0-wei[0])*(1.0-wei[1])*(1.0-wei[2]);
    else if constexpr (wei.size()==2)
      return (1.0-wei[0])*(1.0-wei[1]);
    else if constexpr (wei.size()==1)
      return (1.0-wei[0]);
    else {}
  }

};


template<>
struct index_of_each_node<1,1> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3)
      return std::min(static_cast<idx_type>(idx[0]+1+idx[1]*ng[0]+idx[2]*ng[1]*ng[0]),
                 ng[0]*ng[1]*ng[2]);
    else if constexpr (idx.size()==2)
      return std::min(static_cast<idx_type>(idx[0]+1+idx[1]*ng[0]), ng[0]*ng[1]);
    else if constexpr (idx.size()==1)
      return std::min(static_cast<idx_type>(idx[0]+1),ng[0]);
    else {}
  }
};

template<>
struct weight_to_each_node<1,1> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    if constexpr (wei.size()==3)
      return wei[0]*(1.0-wei[1])*(1.0-wei[2]);
    else if constexpr (wei.size()==2)
      return wei[0]*(1.0-wei[1]);
    else if constexpr (wei.size()==1)
      return wei[0];
    else {}
  }

};

template<>
struct index_of_each_node<1,2> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3)
      return std::min(static_cast<idx_type>(idx[0]+1+(idx[1]+1)*ng[0]+idx[2]*ng[1]*ng[0]),
                 ng[0]*ng[1]*ng[2]);
    else if constexpr (idx.size()==2)
      return std::min(static_cast<idx_type>(idx[0]+1+(idx[1]+1)*ng[0]), ng[0]*ng[1]);
    else {}
  }
};

template<>
struct weight_to_each_node<1,2> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    if constexpr (wei.size()==3)
      return (wei[0])*(wei[1])*(1.0-wei[2]);
    else if constexpr (wei.size()==2)
      return (wei[0])*(wei[1]);
    else {}
  }
};

template<>
struct index_of_each_node<1,3> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3)
      return std::min(static_cast<idx_type>(idx[0]+(idx[1]+1)*ng[0]+idx[2]*ng[1]*ng[0]),
                 ng[0]*ng[1]*ng[2]);
    else if constexpr (idx.size()==2)
      return std::min(static_cast<idx_type>(idx[0]+(idx[1]+1)*ng[0]), ng[0]*ng[1]);
    else {}
  }
};

template<>
struct weight_to_each_node<1,3> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    if constexpr (wei.size()==3)
      return (1.0-wei[0])*(wei[1])*(1.0-wei[2]);
    else if constexpr (wei.size()==2)
      return (1.0-wei[0])*(wei[1]);
    else {}
  }
};


template<>
struct index_of_each_node<1,4> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3)
      return std::min(static_cast<idx_type>(idx[0]+idx[1]*ng[0]+(idx[2]+1)*ng[1]*ng[0]),
                 ng[0]*ng[1]*ng[2]);
    else {}
  }
};

template<>
struct weight_to_each_node<1,4> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    if constexpr (wei.size()==3)
      return (1.0-wei[0])*(1.0-wei[1])*wei[2];
    else {}
  }

};


template<>
struct index_of_each_node<1,5> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3)
      return std::min(static_cast<idx_type>(idx[0]+1+idx[1]*ng[0]+(idx[2]+1)*ng[1]*ng[0]),
                 ng[0]*ng[1]*ng[2]);
    else {}
  }
};

template<>
struct weight_to_each_node<1,5> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    if constexpr (wei.size()==3)
      return wei[0]*(1.0-wei[1])*wei[2];
    else if constexpr (wei.size()==2)
      return wei[0]*(1.0-wei[1]);
    else {}
  }

};


template<>
struct index_of_each_node<1,6> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3)
      return std::min(static_cast<idx_type>(idx[0]+1+(idx[1]+1)*ng[0]+(idx[2]+1)*ng[1]*ng[0]),
                 ng[0]*ng[1]*ng[2]);
    else {}
  }
};

template<>
struct weight_to_each_node<1,6> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    if constexpr (wei.size()==3)
      return (wei[0])*(wei[1])*wei[2];
    else {}
  }
};

template<>
struct index_of_each_node<1,7> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3)
      return std::min(static_cast<idx_type>(idx[0]+(idx[1]+1)*ng[0]+(idx[2]+1)*ng[1]*ng[0]),
                 ng[0]*ng[1]*ng[2]);
    else {}
  }
};

template<>
struct weight_to_each_node<1,7> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    if constexpr (wei.size()==3)
      return (1.0-wei[0])*(wei[1])*wei[2];
    else {}
  }
};


template <typename T>
constexpr T _1D6 = (T) 1./6.;

template <typename T>
constexpr T _2D3 = (T) 2./3.;


template<>
struct index_of_each_node<2,0> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3) {}
    else if constexpr (idx.size()==2) {}
    else if constexpr (idx.size()==1)
      return std::min(static_cast<idx_type>(idx[0]),ng[0]);
    else {}
  }
};
template<>
struct weight_to_each_node<2,0> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    using val_type = decltype(wei)::value_type;
    val_type x = wei[0];
    if constexpr (wei.size()==3) {}
    else if constexpr (wei.size()==2) {}
    else if constexpr (wei.size()==1) 
      return _2D3<val_type>-x*x+0.5*x*x*x;
    else {}
  }

};

template<>
struct index_of_each_node<2,1> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3) {}
    else if constexpr (idx.size()==2) {}
    else if constexpr (idx.size()==1)
      return std::min(static_cast<idx_type>(idx[0]+1),ng[0]);
    else {}
  }
};

template<>
struct weight_to_each_node<2,1> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    using val_type = decltype(wei)::value_type;
    val_type x = 1.0 - wei[0];
    if constexpr (wei.size()==3) {}
    else if constexpr (wei.size()==2) {}
    else if constexpr (wei.size()==1)
      return _2D3<val_type>-x*x+0.5*x*x*x;
    else {}
  }

};


template<>
struct index_of_each_node<2,2> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3) {}
    else if constexpr (idx.size()==2) {}
    else if constexpr (idx.size()==1)
      return std::max(static_cast<idx_type>(0),
                      std::min(static_cast<idx_type>(idx[0]-1),ng[0]));
    else {}
  }
};

template<>
struct weight_to_each_node<2,2> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    using val_type = decltype(wei)::value_type;
    val_type x = 1.0 + wei[0];
    if constexpr (wei.size()==3) {}
    else if constexpr (wei.size()==2) {}
    else if constexpr (wei.size()==1)
     return _1D6<val_type>*(2.-x)*(2.-x)*(2.-x);
    else {}
  }

};

template<>
struct index_of_each_node<2,3> {

  template <typename Array1,typename Array2>
  __host__ __device__
  auto operator()(Array1 idx,Array2 ng) {
    using idx_type = decltype(ng)::value_type;
    if constexpr (idx.size()==3) {}
    else if constexpr (idx.size()==2) {}
    else if constexpr (idx.size()==1)
      return std::min(static_cast<idx_type>(idx[0]+2),ng[0]);
    else {}
  }
};

template<>
struct weight_to_each_node<2,3> {
  
  template <typename Array>
  __host__ __device__
  auto operator()(Array wei) {
    using val_type = decltype(wei)::value_type;
    val_type x = 2.0 - wei[0];
    if constexpr (wei.size()==3) {}
    else if constexpr (wei.size()==2) {}
    else if constexpr (wei.size()==1)
      return _1D6<val_type>*(2.-x)*(2.-x)*(2.-x);
    else {}
  }

};




