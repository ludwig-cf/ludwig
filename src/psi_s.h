/*****************************************************************************
 *
 *  psi_s.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2016 The University of Edinburgh
 *
 *  Contributinf authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef PSI_S_H
#define PSI_S_H

#include <mpi.h>
#include "io_harness.h"
#include "psi.h"
#include "memory.h"

/*
 * We store here the unit charge, the electric permittivity, and the
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
  pe_t * pe;                /* Parallel environment */
  cs_t * cs;                /* Coordinate system */

  int nk;                   /* Number of species */
  int nsites;               /* Number sites storage */
  double * psi;             /* Electric potential */
  double * rho;             /* Charge densities */
  double * diffusivity;     /* Diffusivity for each species */
  int * valency;            /* Valency for each species */
  double e;                 /* unit charge */
  double epsilon;           /* first and reference permittivity */
  double epsilon2;          /* second permittivity */
  double beta;              /* Boltzmann factor (1 / k_B T) */
  double reltol;            /* Relative tolerance for Poisson solver */
  double abstol;            /* Absolute tolerance for Poisson solver */
  int method;               /* Force computation method */
  int maxits;               /* Maximum number of iterations */
  int multisteps;           /* Number of substeps in charge dynamics */
  int skipsteps;            /* Poisson equation solved every skipsteps timesteps */
  int nfreq_io;             /* Field output */
  int nfreq;                /* Residual statisics output */
  double diffacc;           /* Number of substeps in charge dynamics */
  MPI_Datatype psihalo[3];  /* psi field halo */
  MPI_Datatype rhohalo[3];  /* charge densities halo */
  io_info_t * info;         /* I/O informtation */
};

int psi_halo(int nf, double * f, MPI_Datatype halo[3]);

#endif
