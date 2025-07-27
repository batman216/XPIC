#pragma once

template <typename Container>
void half_cell_shift(Container& arr) {

  using val_type = typename Container::value_type;
  thrust::adjacent_difference(arr.begin(),arr.end(),arr.begin(),
                              ::cuda::std::plus<val_type>());
  thrust::for_each(arr.begin(),arr.end(),[]__device__(val_type& val) { val*=0.5; });

}

template <typename Container>
void half_cell_shift(Container& arr1, Container& arr2) {

  using val_type = typename Container::value_type;
  thrust::adjacent_difference(arr1.begin(),arr1.end(),arr2.begin(),
                              ::cuda::std::plus<val_type>());
  thrust::for_each(arr2.begin(),arr2.end(),[]__device__(val_type& val) { val*=0.5; });

}

template <typename Container>
void half_cell_shift_back(Container& arr) {

  using val_type = typename Container::value_type;
  thrust::adjacent_difference(arr.begin()+1,arr.end(),arr.begin(),
                              ::cuda::std::plus<val_type>());
  thrust::for_each(arr.begin(),arr.end(),[]__device__(val_type& val) { val*=0.5; });

}

template <typename Container>
void half_cell_shift_back(Container& arr1,Container& arr2) {

  using val_type = typename Container::value_type;
  thrust::adjacent_difference(arr1.begin()+1,arr1.end(),arr2.begin(),
                              ::cuda::std::plus<val_type>());
  thrust::for_each(arr2.begin(),arr2.end(),[]__device__(val_type& val) { val*=0.5; });

}
template <typename Container>
void smooth(Container& arr,int nl, int nr) {

  using val_type = typename Container::value_type;
  using namespace thrust::placeholders;
  thrust::transform(arr.begin()+nl,arr.end()-nr,arr.begin()+nl+1,arr.begin()+nl,
                    0.5*_1+0.25*_2);
  thrust::transform(arr.begin()+nl,arr.end()-nr,arr.begin()+nl-1,arr.begin()+nl,
                    _1+0.25*_2);

}
template <typename Container>
void periodicBoundaryHalf(Container& arr,int nl, int nr) {

  thrust::copy(arr.begin()+nl, arr.begin()+nl+nr+1, arr.end()-nr-1);
  thrust::copy(arr.end()-nr-nl-1, arr.end()-nr, arr.begin());

}
template <typename Container>
void periodicBoundary(Container& arr,int nl, int nr) {

  thrust::copy(arr.end()-nr-nl-1, arr.end()-nr, arr.begin());
  thrust::copy(arr.begin()+nl, arr.begin()+nl+nr+1, arr.end()-nr-1);

}
template <typename Container>
void setInitField(Container& arr, int n) {

  using val_type = Container::value_type;
  thrust::transform(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(n),
                    arr.begin(), [] __host__ __device__ (int i) {
                      return static_cast<val_type>(std::exp(-(i-3000.0)*(i-3000.0)/30000.0));
                    });

}


template <typename Container>
void periodicBoundaryInterpSum(Container& arr, int nl,int nr) {
      // 处理插值边界
  using val_type = typename Container::value_type;
  using namespace thrust::placeholders;
  thrust::device_vector<val_type> buffer(2);
  thrust::copy(arr.end()-1-nr,arr.end()+1-nr,buffer.begin());
  thrust::transform(arr.begin()+nl-1,
                    arr.begin()+nl+1,//+2 是3阶样条，通用性不足todo
                    arr.end()-2-nr,
                    arr.end()-2-nr,  _1+_2);
  thrust::transform(buffer.begin(), buffer.end(),
                    arr.begin()+nl,
                    arr.begin()+nl, _1+_2);
}


