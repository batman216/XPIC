
#include <fstream>
#include <string>

template <typename PIC>
struct Output {

  using val_type = PIC::value_type;
  std::ofstream ofs; 
  const std::string fname;
  const int mpi_rank;

  template <typename MPI>
  Output(std::string fname,MPI& mpi) 
  : fname(fname), mpi_rank(mpi.world_rank) {
    ofs.open(fname+"@"+std::to_string(mpi_rank)+".out",
             std::ios::binary|std::ios::out);
  }

  template <typename Arr>
  void operator()(const Arr& arr) {

    ofs.write(reinterpret_cast<const char*>(arr.data()),
                                            arr.size()*sizeof(val_type));
  }
  
  ~Output() { ofs.close(); }
  

};
