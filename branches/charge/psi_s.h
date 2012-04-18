/*****************************************************************************
 *
 *  psi_s.h
 *
 *****************************************************************************/

#ifndef PSI_S_H
#define PSI_S_H

#include <mpi.h>
#include "io_harness.h"
#include "psi.h"

struct psi_s {
  int nk;                   /* Number of species */
  double * psi;             /* Electric potential */
  double * rho;             /* Charge densities */
  double * diffusivity;     /* Diffusivity for each species */
  int * valency;            /* Valency for each species */
  MPI_Datatype psihalo[3];  /* psi field halo */
  MPI_Datatype rhohalo[3];  /* charge densities halo */
  struct io_info_t * info;  /* I/O informtation */
};

int psi_halo(int nf, double * f, MPI_Datatype halo[3]);

/* Here for now. psi_ is providing the (opaque) object for the main code. */

extern psi_t * psi_;

#endif
