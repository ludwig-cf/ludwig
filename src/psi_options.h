/*****************************************************************************
 *
 *  psi_options.h
 *
 *  Run time options for electrokinetic sector.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_PSI_OPTIONS_H
#define LUDWIG_PSI_OPTIONS_H

#include "field_options.h"

#define PSI_NKMAX 4

/* Poisson solver method */
typedef enum psi_poisson_solver_enum_s {
  PSI_POISSON_SOLVER_INVALID = 0,
  PSI_POISSON_SOLVER_SOR = 1,
  PSI_POISSON_SOLVER_PETSC = 2
} psi_poisson_solver_enum_t;

/* Force computation method */

typedef enum psi_force_method_enum_s {
  PSI_FORCE_NONE = 0,
  PSI_FORCE_DIVERGENCE,
  PSI_FORCE_GRADMU,
  PSI_FORCE_NTYPES
} psi_force_method_enum_t;

/* There may be a case for splitting this into different concerns;
 * however, it's all here ... */

typedef struct psi_options_s psi_options_t;

struct psi_options_s {

  int nk;                         /* Number of charged species */

  /* Physics */
  double e;                       /* Unit charge */
  double beta;                    /* Boltzmann factor (1 / k_B T) */
  double epsilon1;                /* First permittivity */
  double epsilon2;                /* Second permittivity (if required) */
  double e0[3];                   /* External electric field */
  double diffusivity[PSI_NKMAX];  /* Per species diffusivity */
  int    valency[PSI_NKMAX];      /* Per species charge valency */

  /* Poisson solver (iterative) */
  int psolver;                    /* Poisson solver id */
  int maxits;                     /* Maximum iterations in solver */
  int nfreq;                      /* Frequency of information (iterations) */
  double reltol;                  /* Relative tolerance */
  double abstol;                  /* Absolute tolerance */

  /* Time stepping for Nernst Planck */
  int nsolver;                    /* Nernst Planck method */
  int nsmallstep;                 /* No. small timesteps in time splitting */
  double diffacc;                 /* FIXME: what exactly is this? */

  /* Other */
  int method;                     /* Force computation method */
  field_options_t psi;            /* Field options for potential (i/o etc) */
  field_options_t rho;            /* field options for charges   (i/o etc) */
};

psi_options_t psi_options_default(int nhalo);


int psi_bjerrum_length1(const psi_options_t * opts, double * lb);
int psi_bjerrum_length2(const psi_options_t * opts, double * lb);
int psi_debye_length1(const psi_options_t * obj, double rho_b, double * ld);
int psi_debye_length2(const psi_options_t * obj, double rho_b, double * ld);

#endif
