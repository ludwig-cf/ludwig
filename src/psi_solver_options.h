/*****************************************************************************
 *
 *  psi_solver_options.h
 *
 *  Poisson solver options.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_PSI_SOLVER_OPTIONS_H
#define LUDWIG_PSI_SOLVER_OPTIONS_H

#include "util_json.h"

/* Poisson solver method */

typedef enum psi_poisson_solver_enum_s {
  PSI_POISSON_SOLVER_INVALID = 0,
  PSI_POISSON_SOLVER_SOR = 1,
  PSI_POISSON_SOLVER_PETSC = 2,
  PSI_POISSON_SOLVER_NONE = 3
} psi_poisson_solver_enum_t;

/* This is intended to be general; some components might not be relevant
   in all specific cases. */

typedef struct psi_solver_options_s psi_solver_options_t;

struct psi_solver_options_s {

  psi_poisson_solver_enum_t psolver;   /* Poisson solver id */
  int maxits;                          /* Maximum iterations in solver */
  int verbose;                         /* Level of verbosity */
  int nfreq;                           /* Frequency of report */
  int nstencil;                        /* Stencil option */

  double reltol;                       /* Relative tolerance */
  double abstol;                       /* Absolute tolerance */
};

const char * psi_poisson_solver_to_string(psi_poisson_solver_enum_t mytype);
psi_poisson_solver_enum_t psi_poisson_solver_from_string(const char * str);

psi_solver_options_t psi_solver_options_default(void);
psi_solver_options_t psi_solver_options_type(psi_poisson_solver_enum_t mytype);

int psi_solver_options_to_json(const psi_solver_options_t * opts, cJSON ** js);
int psi_solver_options_from_json(const cJSON * json, psi_solver_options_t * p);

#endif
