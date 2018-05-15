/*****************************************************************************
 *
 *  free_energy.h
 *
 *  The 'abstract' free energy interface.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2018 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_FREE_ENERGY_H
#define LUDWIG_FREE_ENERGY_H

#include "memory.h"

enum fe_id_enum {FE_SYMMETRIC,
		 FE_BRAZOVSKII,
		 FE_POLAR,
		 FE_LC,
		 FE_ELECTRO,
		 FE_ELECTRO_SYMMETRIC,
		 FE_LC_DROPLET};

typedef struct fe_s fe_t;
typedef struct fe_vt_s fe_vt_t;

typedef int (* fe_free_ft)(fe_t * fe);
typedef int (* fe_id_ft)(fe_t * fe);
typedef int (* fe_target_ft)(fe_t * fe, fe_t ** target);
typedef int (* fe_fed_ft)(fe_t * fe, int index, double * fed);
typedef int (* fe_mu_ft)(fe_t * fe, int index, double * mu);
typedef int (* fe_str_ft)(fe_t * fe, int index, double s[3][3]);
typedef int (* fe_mu_solv_ft)(fe_t * fe, int index, int k, double * mu);
typedef int (* fe_hvector_ft)(fe_t * fe, int index, double h[3]);
typedef int (* fe_htensor_ft)(fe_t * fe, int index, double h[3][3]);
/* Vectorised versions */
typedef void (* fe_htensor_v_ft)(fe_t * fe, int index,double h[3][3][NSIMDVL]);
typedef void (* fe_stress_v_ft)(fe_t * fe, int index, double s[3][3][NSIMDVL]);

struct fe_vt_s {
  /* Order is important: actual tables must appear thus... */
  fe_free_ft free;              /* Virtual destructor */
  fe_target_ft target;          /* Return target pointer cast to fe_t */
  fe_fed_ft fed;                /* Freee energy density */
  fe_mu_ft mu;                  /* Chemical potential */
  fe_mu_solv_ft mu_solv;        /* Solvation chemical potential */
  fe_str_ft stress;             /* Chemical stress (total) */
  fe_str_ft str_symm;           /* Symmetric stress */
  fe_str_ft str_anti;           /* Antisymmetric stress */
  fe_hvector_ft hvector;        /* Vector molecular field */
  fe_htensor_ft htensor;        /* Tensor molecular field */
  fe_htensor_v_ft htensor_v;    /* Vectorised version */
  fe_stress_v_ft stress_v;      /* Vectorised stress (total) version */
  fe_stress_v_ft str_symm_v;    /* Symmetric part */
  fe_stress_v_ft str_anti_v;    /* Antisymmetric part */
};

struct fe_s {
  fe_vt_t * func;
  int id;
  int use_stress_relaxation;
};

#endif
