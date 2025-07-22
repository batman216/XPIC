
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

  Interp<PIC> interp(cell,mpi);

  Maxwell<PIC> maxwell(cell,mpi);
  Boris<PIC> push;
  //setInitField(cell.efield[2],cell.ng1);

  diag::TraceParticle<PIC> trace(mpi);
  int step = -1;

  std::cout << "dt=" << cell.dt << std::endl;  
  while (++step<=12000) {

    auto start = std::chrono::high_resolution_clock::now(); 

    maxwell(cell);
    interp.calWeight(species,cell);

    auto m1 = std::chrono::high_resolution_clock::now(); 
    interp.particle2cell(species,cell);
    auto m2 = std::chrono::high_resolution_clock::now(); 
    interp.cell2particle(species,cell);
    auto md = std::chrono::duration_cast<std::chrono::milliseconds>(m2 - m1);

    if(mpi.world_rank==0) std::cout  << " field("<< md.count() <<"ms), ";
    if (mpi.world_rank==0&&step%200==0) {
      Output<PIC> oute3("e3_"+std::to_string(step),mpi);
      thrust::host_vector<double> he3 = cell.efield[2];
      oute3(he3);
      Output<PIC> outd("d_"+std::to_string(step),mpi);
      thrust::host_vector<double> hd = cell.edens[0];
      outd(hd);
      Output<PIC> outj1("j1_"+std::to_string(step),mpi);
      thrust::host_vector<double> hj1 = cell.jsfield[0][0];
      outj1(hj1);
      Output<PIC> outj2("j2_"+std::to_string(step),mpi);
      thrust::host_vector<double> hj2 = cell.jfield[1];
      outj2(hj2);
      Output<PIC> outj3("j3_"+std::to_string(step),mpi);
      thrust::host_vector<double> hj3 = cell.jfield[2];
      outj3(hj3);
      Output<PIC> oute1("e1_"+std::to_string(step),mpi);
      thrust::host_vector<double> he1 = cell.efield[0];
      oute1(he1);
      Output<PIC> oute2("e2_"+std::to_string(step),mpi);
      thrust::host_vector<double> he2 = cell.efield[1];
      oute2(he2);
/*
      Output<PIC> outb1("b1_"+std::to_string(step),mpi);
      thrust::host_vector<double> hb1 = cell.bfield[0];
      outb1(hb1);
      Output<PIC> outb2("b2_"+std::to_string(step),mpi);
      thrust::host_vector<double> hb2 = cell.bfield[1];
      outb2(hb2);
      Output<PIC> outb3("b3_"+std::to_string(step),mpi);
      thrust::host_vector<double> hb3 = cell.bfield[2];
      outb3(hb3);
      */
    }
    if (mpi.world_rank==0&&step%1==0) {
      /*
      Output<PIC> outv1("v1_"+std::to_string(step),mpi);
      thrust::host_vector<double> hv1 = species[0].E[0];
      outv1(hv1);
      Output<PIC> outv2("v2_"+std::to_string(step),mpi);
      thrust::host_vector<double> hv2 = species[0].v[1];
      outv2(hv2);
      Output<PIC> outv3("v3_"+std::to_string(step),mpi);
      thrust::host_vector<double> hv3 = species[0].v[2];
      outv3(hv3);
      Output<PIC> outv("v_"+std::to_string(step),mpi);
      thrust::host_vector<double> hv = species[0].v[0];
      outv(hv);

      Output<PIC> outt("p_"+std::to_string(step),mpi);
      thrust::host_vector<double> hp = species[0].x[0];
      outt(hp);
      */
    }

    push(species,cell);
    trace(species[0]);

    if constexpr (PIC::DVD) swap(species);


    auto end = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if(mpi.world_rank==0) std::cout << "step" << step  << "("<< duration.count() <<"ms)"<< std::endl;  
    cudaDeviceSynchronize();
  }
}
