/*****************************************************************************
 *
 *  psi_solver.h
 *
 *  Abstract Poisson solver interface.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_PSI_SOLVER_H
#define LUDWIG_PSI_SOLVER_H

#include "free_energy.h"
#include "psi.h"

typedef struct psi_solver_vt_s psi_solver_vt_t;
typedef struct psi_solver_s psi_solver_t;

typedef int (* psi_solver_free_ft)  (psi_solver_t ** psolver);
typedef int (* psi_solver_solve_ft) (psi_solver_t * solver, int ntimestep);

struct psi_solver_vt_s {
  psi_solver_free_ft      free;   /* Destructor */
  psi_solver_solve_ft     solve;  /* Driver of solve */
};

struct psi_solver_s {
  const psi_solver_vt_t * impl;   /* Implementation vtable */
};

/* Factory methods */

int psi_solver_create(psi_t * psi, psi_solver_t ** solver);

/* For dielectric contrast, we need some abstract component which
 * involves the free energy. However, as only electrosymmetric is
 * relevant at the moment, I haven't added a further method to the
 * abstract free energy interface. So we have a slightly add hoc
 * addition here... */

typedef struct var_epsilon_s var_epsilon_t;
typedef int (* var_epsilon_ft)(void * fe, int index, double * epsilon);

struct var_epsilon_s {
  fe_t * fe;
  var_epsilon_ft epsilon;
};

int psi_solver_var_epsilon_create(psi_t * psi, var_epsilon_t epsilon,
				  psi_solver_t ** solver);
#endif
