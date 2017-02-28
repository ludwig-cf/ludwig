/*****************************************************************************
 *
 *  fe_electro_symmetric.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef FE_ELECTRO_SYMMETRIC_H
#define FE_ELECTRO_SYMMETRIC_H

#include "free_energy.h"
#include "symmetric.h"
#include "fe_electro.h"
#include "psi.h"

typedef struct fe_electro_symmetric_s fe_es_t;
typedef struct fe_electro_symmetric_param_s fe_es_param_t;

struct fe_electro_symmetric_param_s {

  double epsilon1;            /* Dielectric constant phase 1 */
  double epsilon2;            /* Dielectric constant phase 2 */
  double epsilonbar;          /* Mean dielectric */
  double gamma;               /* Dielectric contrast */

  double deltamu[PSI_NKMAX];  /* Solvation free energy difference [species] */
  int nk;                     /* Number of species - same os psi_nk() */
};

__host__ int fe_es_create(pe_t * pe, cs_t * cs, fe_symm_t * fe_symm,
			  fe_electro_t * fe_elec,
			  psi_t * psi, fe_es_t ** fe);
__host__ int fe_es_free(fe_es_t * fe);
__host__ int fe_es_target(fe_es_t * fe, fe_t ** target);

__host__
int fe_es_mu_ion_solv(fe_es_t * fe, int index, int n, double * mu);

__host__
int fe_es_deltamu_set(fe_es_t * fe, int nk, double * deltamu);

__host__
int fe_es_epsilon_set(fe_es_t * fe, double e1, double e2);

__host__
int fe_es_var_epsilon(fe_es_t * fe, int index, double * epsilon);

__host__
int fe_es_fed(fe_es_t * fe, int index, double * fed);

__host__
int fe_es_mu_phi(fe_es_t * fe, int index, double * mu);

__host__
int fe_es_stress_ex(fe_es_t * fe, int index, double s[3][3]);

#endif
