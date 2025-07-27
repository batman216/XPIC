#!/bin/bash

NSYS=0
EXEC=bld/xpic
ARGS=()
NP=${1:-1}
if [ "$2" == "--nsys" ]; then
  NSYS=1
fi

if [ "$NSYS" -eq 1 ]; then
  mpirun -np $NP bash -c \
    'nsys profile -o my_report_rank$OMPI_COMM_WORLD_RANK ./'"$EXEC"''
else
  mpirun -np $NP "$EXEC" 
fi


