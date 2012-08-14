/*****************************************************************************
 *
 *  phi_stats.h
 *
 *  $Id: phi_stats.h,v 1.5 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2008)
 *
 *****************************************************************************/

#ifndef PHI_STATS_
#define PHI_STATS_

#include <mpi.h>
#include <field.h>

void phi_set_mean_phi(double);
void phi_stats_print_stats(void);
void phi_init_block(const double xi0);
void phi_init_bath(void);
void phi_init_surfactant(double);

int stats_field_info(field_t * obj);
int stats_field_reduce(field_t * obj, double * fmin, double * fmax,
		       double * fsum, double * fvar, double * fvol,
		       int rank, MPI_Comm comm);
int stats_field_local(field_t * obj, double * fmin, double * fmax,
		      double * fsum, double * fvar, double * fvol);
#endif
