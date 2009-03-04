#ifndef LCPARALLEL_HH
#define LCPARALLEL_HH

#ifdef PARALLEL
#include <mpi.h>
#endif

#include <fstream.h>
#include <math.h>
#include <stdlib.h>
#include <iomanip.h>

void exchangeMomentumAndQTensor(void);
void exchangeTau(void);
void communicateOldDistributions(double ****fold);

#endif
