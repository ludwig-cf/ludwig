/*****************************************************************************
 *
 *  psi.h
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

#ifndef PSI_H
#define PSI_H

#include "pe.h"
#include "coords.h"
#include "io_harness.h"
#include "map.h"

/* PSI_NKMAX is here to allow us to declare arrays to hold
 * per-species quantities. It avoids allocation of  short arrays,
 * which is slightly tedious, particularly on the device. */

#define PSI_NKMAX 4

/* Force computation method */

enum psi_force_method {PSI_FORCE_NONE = 0,
		       PSI_FORCE_DIVERGENCE,
		       PSI_FORCE_GRADMU,
		       PSI_FORCE_NTYPES
};

typedef struct psi_s psi_t;

/* f_vare_t describes the signature of the function expected
 * to return the permittivity as a function of position index. */
typedef int (* f_vare_t)(void * fe, int index, double * epsilon);

int psi_create(pe_t * pe, cs_t * cs, int nk, psi_t ** pobj);
void psi_free(psi_t * obj);
int psi_init_io_info(psi_t * obj, int grid[3], int form_in, int form_out);
int psi_io_info(psi_t * obj, io_info_t ** info);

int psi_nk(psi_t * obj, int * nk);
int psi_valency(psi_t * obj, int n, int * iv);
int psi_valency_set(psi_t * obj, int n, int iv);
int psi_diffusivity(psi_t * obj, int n, double * diff);
int psi_diffusivity_set(psi_t * obj, int n, double diff);
int psi_halo_psi(psi_t * obj);
int psi_halo_psijump(psi_t * obj);
int psi_halo_rho(psi_t * obj);

int psi_rho(psi_t * obj, int index, int n, double * rho);
int psi_rho_set(psi_t * obj, int index, int n, double rho);
int psi_psi(psi_t * obj, int index, double * psi);
int psi_psi_set(psi_t * obj, int index, double psi);
int psi_rho_elec(psi_t * obj, int index, double * rho_elec);
int psi_unit_charge(psi_t * obj, double * eunit);
int psi_unit_charge_set(psi_t * obj, double eunit); 
int psi_beta(psi_t * obj, double * beta);
int psi_beta_set(psi_t * obj, double beta);
int psi_epsilon(psi_t * obj, double * epsilon);
int psi_epsilon_set(psi_t * obj, double epsilon);
int psi_epsilon2(psi_t * obj, double * epsilon2);
int psi_epsilon2_set(psi_t * obj, double epsilon2);
int psi_ionic_strength(psi_t * psi, int index, double * sion);
int psi_bjerrum_length(psi_t * obj, double * lb);
int psi_bjerrum_length2(psi_t * obj, double * lb);
int psi_debye_length(psi_t * obj, double rho_b, double * ld);
int psi_debye_length2(psi_t * obj, double rho_b, double * ld);
int psi_surface_potential(psi_t * obj, double sigma, double rho_b,
			  double * sp);
int psi_reltol(psi_t * obj, double * reltol);
int psi_abstol(psi_t * obj, double * abstol);
int psi_maxits(psi_t * obj, int * maxits);
int psi_reltol_set(psi_t * obj, double reltol);
int psi_abstol_set(psi_t * obj, double abstol);
int psi_maxits_set(psi_t * obj, int maxits);
int psi_nfreq_set(psi_t * psi, int nfreq);
int psi_output_step(psi_t * psi);

int psi_multisteps(psi_t * obj, int * multisteps);
int psi_multisteps_set(psi_t * obj, int multisteps);
int psi_multistep_timestep(psi_t * obj, double * dt);
int psi_diffacc(psi_t * obj, double * diffacc);
int psi_diffacc_set(psi_t * obj, double diffacc);
int psi_skipsteps(psi_t * obj);
int psi_skipsteps_set(psi_t * obj, double skipsteps);
int psi_zero_mean(psi_t * obj);
int psi_force_method(psi_t * obj, int * flag);
int psi_force_method_set(psi_t * obj, int flag);

int psi_electroneutral(psi_t * obj, map_t * map);
#endif
