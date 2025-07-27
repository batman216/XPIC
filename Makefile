EXE = xpic
CXX = nvcc

NVCFlAG = --expt-relaxed-constexpr --extended-lambda -arch=sm_89
CXXFLAG = -std=c++20 -Xcompiler 
LDFLAG  = -Isrc/include -I${CUMATH_HOME}/include -L${CUMATH_HOME}/lib \
					-I${NCCL_HOME}/include -L${NCCL_HOME}/lib     \
					-I${MPI_HOME}/include -L${MPI_HOME}/lib     \
          -lmpi -lnccl -lcusparse -lcuda -lcudart

ifeq ($(mode),debug)
	CXXFLAG += -g
endif

SRC = src
BLD = bld
RUN = run
ILD = include

INPUT = xpic.in

$(shell mkdir -p ${BLD})


CPP = ${wildcard ${SRC}/*.cpp}
CU  = ${wildcard ${SRC}/*.cu}
HPP = ${wildcard ${ILD}/*.hpp}

CUOBJ = ${patsubst ${SRC}/%.cu,${BLD}/%.o,${CU}}
CPPOBJ = ${patsubst ${SRC}/%.cpp,${BLD}/%.o,${CPP}}


${BLD}/${EXE}: ${CUOBJ} ${CPPOBJ}
	${CXX} $^ ${NVCFlAG} ${LDFLAG} -o $@

${BLD}/%.o: ${SRC}/%.cu 
	${CXX} ${CXXFLAG} ${LDFLAG} ${NVCFlAG} -c $< -o $@

${BLD}/%.o: ${SRC}/%.cpp 
	${CXX} ${CXXFLAG} ${LDFLAG} ${NVCFlAG} -c $< -o $@


run: ${BLD}/${EXE} ${INPUT}
	mkdir -p ${RUN} && cp $^ ${RUN} && cd ${RUN} && ./${EXE}

clean:
	rm -rf ${BLD}

show:
	echo ${CPP} ${OBJ}
