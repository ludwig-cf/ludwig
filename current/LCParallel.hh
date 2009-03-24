#ifndef LCPARALLEL_HH
#define LCPARALLEL_HH

#ifdef PARALLEL
#include <mpi.h>
#endif

#include <math.h>
#include <stdlib.h>

#include <fstream>
#include <iomanip>

void exchangeMomentumAndQTensor(void);
void exchangeTau(void);
void communicateOldDistributions(double ****fold);

#endif
