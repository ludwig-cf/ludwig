/*****************************************************************************
 *
 *  psi_sor.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Contributing Authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Ignacio Pagonabarraga (ipagonabarraga@ub.edu)
 *
 *  (c) 2012-2023 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_PSI_SOLVER_SOR_H
#define LUDWIG_PSI_SOLVER_SOR_H

#include "psi_solver.h"

typedef struct psi_solver_sor_s psi_solver_sor_t;

struct psi_solver_sor_s {
  psi_solver_t super;                    /* superclass block */
  psi_t * psi;                           /* Reference to psi structure */
  fe_t * fe;                             /* abstract free energy */
  var_epsilon_ft epsilon;                /* provides local epsilon */
};

int psi_solver_sor_create(psi_t * psi, psi_solver_sor_t ** sor);
int psi_solver_sor_free(psi_solver_sor_t ** sor);
int psi_solver_sor_solve(psi_solver_sor_t * sor, int ntimestep);

/* This might actually be a separate solver type. */

int psi_solver_sor_var_epsilon_create(psi_t * psi, var_epsilon_t epsilon,
				      psi_solver_sor_t ** sor);

int psi_solver_sor_var_epsilon_solve(psi_solver_sor_t * sor, int ntimestep);


#endif
