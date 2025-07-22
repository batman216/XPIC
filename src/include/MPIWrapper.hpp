#pragma once
#include <mpi.h>

#include <nccl.h>

uint64_t getHostHash(const char *string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

void getHostName(char *hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

static bool mpi_finalized = false;
struct MPIWrapper {

  int world_size, world_rank, local_rank=0;
  char flag;

  ncclUniqueId nccl_id;
  cudaStream_t s;
  ncclComm_t comm;

  ~MPIWrapper() {
    if (!mpi_finalized) {
      MPI_Finalize();
      mpi_finalized = true;
    }
  }
  MPIWrapper() {
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);  
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);  

    /// determine the flag of the GPU, 'left','right',or 'middle'
    if (world_rank == 0) flag = 'l';
    else if (world_rank == world_size-1) flag = 'r';
    else flag = 'm';

    uint64_t host_hashs[world_size];
    char hostname[1024];
    getHostName(hostname, 1024);
    host_hashs[world_rank] = getHostHash(hostname);
    MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,host_hashs,
                  sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
    for (int p=0; p<world_size; p++) {
      if (p==world_rank) break;
      if (host_hashs[p]==host_hashs[world_rank]) local_rank++;
    }

    if (world_rank==0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast((void*)&nccl_id,sizeof(nccl_id),MPI_BYTE,0,MPI_COMM_WORLD);
  
    cudaSetDevice(local_rank);
    cudaStreamCreate(&s);
    ncclCommInitRank(&comm,world_size,nccl_id,world_rank);

  }

};

template <typename val_type> 
struct nccl_traits {
  ncclDataType_t name;
  nccl_traits() {
    if constexpr (std::is_same<val_type,float>::value) 
      name = ncclFloat;
    else if constexpr(std::is_same<val_type,double>::value)
      name = ncclDouble;
    else {}

  }
  
};


