
#include <string>

#include "ParticleInCell.hpp"
#include "LoadParticle.hpp"
#include "MemAlloc.hpp"
#include "Interp.hpp"
#include "ParticleSwap.hpp"
#include "Input.hpp"
#include "Diagnosis.hpp"
#include "Output.hpp"
#include "MPIWrapper.hpp"
#include "Maxwell.hpp"

#include "particle_algorithm/Boris.hpp"

#include "thrust/sequence.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/copy.h"

using PIC = ParticleInCell<std::size_t,double, 
                           thrust::device_vector,
                           thrust::device_vector,  
                           false,
                           2, /* particle species*/
                           1, /* x dimension */
                           3, /* v dimension */
                           2  /* inerp order */>;
