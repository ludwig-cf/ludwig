/*****************************************************************************
 *
 *  psi.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_PSI_H
#define LUDWIG_PSI_H

#include "pe.h"
#include "field.h"
#include "map.h"
#include "psi_options.h"
#include "stencil.h"

typedef struct psi_s psi_t;

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

  field_t * psi;            /* Electric potential */
  field_t * rho;            /* Charge densities */

  double * diffusivity;     /* Diffusivity for each species */
  int * valency;            /* Valency for each species */
  double e;                 /* unit charge */
  double epsilon;           /* first and reference permittivity */
  double epsilon2;          /* second permittivity */
  double beta;              /* Boltzmann factor (1 / k_B T) */

  int method;               /* Force computation method */
  int multisteps;           /* Number of substeps in charge dynamics */
  int nfreq_io;             /* Field output */

  double diffacc;           /* Number of substeps in charge dynamics */
  double e0[3];             /* External electric field */

  /* Solver options */
  psi_solver_options_t solver;      /* User options */
  stencil_t * stencil;              /* Finite difference stencil info */

};


int psi_create(pe_t * pe, cs_t * cs, const psi_options_t * opts, psi_t ** p);
int psi_free(psi_t ** psi);

int psi_initialise(pe_t * pe, cs_t * cs, const psi_options_t * opts,
		   psi_t * psi);
int psi_finalise(psi_t * psi);

int psi_nk(psi_t * obj, int * nk);
int psi_valency(psi_t * obj, int n, int * iv);
int psi_diffusivity(psi_t * obj, int n, double * diff);
int psi_halo_psi(psi_t * obj);
int psi_halo_psijump(psi_t * obj);
int psi_halo_rho(psi_t * obj);

int psi_rho(psi_t * obj, int index, int n, double * rho);
int psi_rho_set(psi_t * obj, int index, int n, double rho);
int psi_psi(psi_t * obj, int index, double * psi);
int psi_psi_set(psi_t * obj, int index, double psi);
int psi_rho_elec(psi_t * obj, int index, double * rho_elec);
int psi_unit_charge(psi_t * obj, double * eunit);
int psi_beta(psi_t * obj, double * beta);
int psi_epsilon(psi_t * obj, double * epsilon);
int psi_epsilon2(psi_t * obj, double * epsilon2);
int psi_ionic_strength(psi_t * psi, int index, double * sion);
int psi_surface_potential(psi_t * obj, double sigma, double rho_b,
			  double * sp);
int psi_reltol(psi_t * obj, double * reltol);
int psi_abstol(psi_t * obj, double * abstol);
int psi_maxits(psi_t * obj, int * maxits);
int psi_output_step(psi_t * psi, int its);

int psi_multisteps(psi_t * obj, int * multisteps);
int psi_multistep_timestep(psi_t * obj, double * dt);
int psi_diffacc(psi_t * obj, double * diffacc);
int psi_zero_mean(psi_t * obj);
int psi_force_method(psi_t * obj, int * flag);
int psi_force_method_set(psi_t * obj, int flag);

int psi_electroneutral(psi_t * obj, map_t * map);

#endif
