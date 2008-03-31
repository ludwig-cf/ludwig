#ifndef LCPARALLEL_HH
#define LCPARALLEL_HH

#ifdef PARALLEL
#include <mpi.h>
#endif

#include <fstream.h>
#include <math.h>
#include <stdlib.h>
#include <iomanip.h>

#define LCMAIN_HH
#include "LCMain_hybrid.hh"
#include "String.hh"

void exchangeMomentumAndQTensor();
void communicateOldDistributions(double ****fold);

#endif
