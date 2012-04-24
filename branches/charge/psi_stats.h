/*****************************************************************************
 *
 *  psi_stats.h
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

#ifndef PSI_STATS_H
#define PSI_STATS_H

#include <mpi.h>
#include "psi.h"

int psi_stats_info(psi_t * psi);
int psi_stats_local(psi_t * psi, double * rho_min, double * rho_max,
		    double * rho_tot);
int psi_stats_reduce(psi_t * psi, double * rho_min, double * rho_max,
		     double * rho_tot, int rank, MPI_Comm comm);
#endif
