
#include "include/headers.hpp"

int main(/*Tian-Xing Hu*/) {

  cudaDeviceReset();
  MPIWrapper mpi;

  Species<PIC> species;  Cell<PIC> cell;

  Input<PIC,MPIWrapper> in(mpi); 
  MemAlloc(species,cell,in);

  std::cout << "MemAlloc success." << std::endl;

  Swap<PIC> swap(cell,mpi);

  LoadParticle<PIC,Input<PIC,MPIWrapper>> load(in);
  load(species);
  std::cout << "LoadParticle success." << std::endl;
  int step = -1;

  Interp<PIC> interp(cell,mpi);
  interp.calWeight(species,cell);
  interp.particle2cell(species,cell); // rho[-1/2], J[-1/2]

//  setInitField(cell.efield[2],cell.ng1);
  Maxwell<PIC> maxwell(cell,mpi);
  maxwell(cell);
  interp.cell2particle(species,cell);

  Boris<PIC> push;
  push(species,cell);  // rho[1/2],J[1/2]

  diag::TraceParticle<PIC> trace(mpi);

  Output<PIC> out(in,mpi,species,cell);
  while (++step<=in.stopat) {
    auto start = std::chrono::high_resolution_clock::now(); 

    out(step,species,cell);
    trace(species[0]);

    interp.calWeight(species,cell);
    interp.particle2cell(species,cell);
    calVBCurrent<PIC>(cell);
    interp.cell2particle(species,cell);
//    maxwell(cell);

    push(species,cell);

    if constexpr (PIC::DVD) swap(species);


    auto end = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if(mpi.world_rank==0) std::cout << "step" << step  << "("<< duration.count() <<"ms)"<< std::endl;  
    cudaDeviceSynchronize();
  }
}
