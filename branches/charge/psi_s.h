/*****************************************************************************
 *
 *  psi_s.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PSI_S_H
#define PSI_S_H

#include <mpi.h>
#include "io_harness.h"
#include "psi.h"

/*
 * We store here the unit charge, the electric permeativity, and the
 * temperature, all in lattice units. This allows us to work out the
 * Bjerrum length,
 *   l_B = e^2 / (4 \pi epsilon_0 epsilon_r kT)
 * which is the scale at which the interaction energy between two
 * unit charges is equal to kT. The aim is that the Bjerrum length
 * should be < 1 (e.g. 0.7) in lattice units.
 *
 * For water at room temperature, the Bjerrum length is about
 * 7 Angstrom.
 *
 */

struct psi_s {
  int nk;                   /* Number of species */
  double * psi;             /* Electric potential */
  double * rho;             /* Charge densities */
  double * diffusivity;     /* Diffusivity for each species */
  int * valency;            /* Valency for each species */
  double e;                 /* unit charge */
  double epsilon;           /* reference permeativity */
  double beta;              /* Boltzmann factor (1 / k_B T) */
  MPI_Datatype psihalo[3];  /* psi field halo */
  MPI_Datatype rhohalo[3];  /* charge densities halo */
  struct io_info_t * info;  /* I/O informtation */
};

int psi_halo(int nf, double * f, MPI_Datatype halo[3]);

/* Here for now. psi_ is providing the (opaque) object for the main code. */

extern psi_t * psi_;

#endif
