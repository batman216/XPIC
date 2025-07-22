#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>

#define TGET(x,i) thrust::get<i>(x)

template <typename PIC>
struct Boris {

  using Container = typename PIC::ParticleContainer;
  using val_type  = PIC::value_type;
  static constexpr std::size_t dim_x = PIC::dim_x, 
                               dim_v = PIC::dim_v;
  using Tuple_v = thrust::detail::
                  tuple_of_iterator_references<
                  val_type&,val_type&,val_type&>;
  using Tuple_vv = thrust::detail::
                        tuple_of_iterator_references
                          <Tuple_v,Tuple_v,Tuple_v>;

  using ValItor    = thrust::device_vector<val_type>::iterator;
  using ZipItor_v  = thrust::zip_iterator<thrust::tuple<ValItor,ValItor,ValItor>>;

  ZipItor_v  vzip, bzip, ezip;
  Boris() {}

  template <typename Species,typename Cell>
  void operator()(Species& sp, Cell& cell){
      
    for (int s=0; s<PIC::n_species; ++s) {
      ezip = thrust::make_zip_iterator(sp[s].E[0].begin(),
                                       sp[s].E[1].begin(),
                                       sp[s].E[2].begin());
      bzip = thrust::make_zip_iterator(sp[s].B[0].begin(),
                                       sp[s].B[1].begin(),
                                       sp[s].B[2].begin());


      vzip = thrust::make_zip_iterator(sp[s].v[0].begin(),
                                       sp[s].v[1].begin(),
                                       sp[s].v[2].begin());
      val_type dt = cell.dt;
      val_type B0[3] = {0,0,0};
      val_type E0[3] = {0,0,0};
      auto np = sp[s].v[0].size();
      auto qdm_dt = dt*sp[s].qdm*.5;
      val_type c = PIC::Constant::c;

      auto vvzip = thrust::make_zip_iterator(vzip,ezip,bzip);
      thrust::for_each(vvzip,vvzip+np,[c,qdm_dt,E0,B0]__host__ __device__(Tuple_vv t)
                      {
                         val_type v0[dim_v], ee[dim_v],ss[dim_v], tt[dim_v];
                         v0[0] = TGET(TGET(t,0),0)+qdm_dt*(E0[0]+TGET(TGET(t,1/*E*/),0)); 
                         v0[1] = TGET(TGET(t,0),1)+qdm_dt*(E0[1]+TGET(TGET(t,1/*E*/),1)); 
                         v0[2] = TGET(TGET(t,0),2)+qdm_dt*(E0[2]+TGET(TGET(t,1/*E*/),2)); 

                         ee[0] = (E0[0]+TGET(TGET(t,1/*E*/),0))*qdm_dt;
                         ee[1] = (E0[1]+TGET(TGET(t,1/*E*/),1))*qdm_dt;
                         ee[2] = (E0[2]+TGET(TGET(t,1/*E*/),2))*qdm_dt;
                         tt[0] = (B0[0]+TGET(TGET(t,2/*B*/),0))*qdm_dt/c;
                         tt[1] = (B0[1]+TGET(TGET(t,2/*B*/),1))*qdm_dt/c;
                         tt[2] = (B0[2]+TGET(TGET(t,2/*B*/),2))*qdm_dt/c;
                         
                         for (int i=0; i<3; ++i) 
                           ss[i] = 2.0*tt[i]/(1.0+tt[0]*tt[0]+tt[1]*tt[1]+tt[2]*tt[2]);

                         TGET(TGET(t,0),0) = v0[0] + ee[0]
                                           + (v0[1]+v0[2]*tt[0]-v0[0]*tt[2])*ss[2]
                                           - (v0[2]+v0[0]*tt[1]-v0[1]*tt[0])*ss[1];
                         TGET(TGET(t,0),1) = v0[1] + ee[1]
                                           + (v0[2]+v0[0]*tt[1]-v0[1]*tt[0])*ss[0]
                                           - (v0[0]+v0[1]*tt[2]-v0[2]*tt[1])*ss[2];
                         TGET(TGET(t,0),2) = v0[2] + ee[2]
                                           + (v0[0]+v0[1]*tt[2]-v0[2]*tt[1])*ss[1]
                                           - (v0[1]+v0[2]*tt[0]-v0[0]*tt[2])*ss[0];
                       });

      val_type L = cell.L[0], a = cell.a[0], b = cell.b[0];
      thrust::transform(sp[s].x[0].begin(),sp[s].x[0].end(),sp[s].v[0].begin(),sp[s].x[0].begin(),
                        [dt]__host__ __device__(val_type x, val_type v)
                       {
                          return x + v*dt;
                       });
      thrust::transform_if(sp[s].x[0].begin(),sp[s].x[0].end(),sp[s].x[0].begin(),
                           [L]__host__ __device__(val_type& val) 
                           { return val + L; },
                           [a]__host__ __device__(val_type& val) 
                           { return val < a; });
      thrust::transform_if(sp[s].x[0].begin(),sp[s].x[0].end(),sp[s].x[0].begin(),
                           [L]__host__ __device__(val_type& val) 
                           { return val - L; },
                           [b]__host__ __device__(val_type& val) 
                           { return val > b; });
      

    }
  }
};
