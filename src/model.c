/*****************************************************************************
 *
 *  model.c
 *
 *  This is a wrapper for the lattice Boltzmann model:
 *  either _D3Q15_ or _D3Q19_.
 *
 *  $Id: model.c,v 1.7 2006-10-12 14:09:18 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "runtime.h"
#include "coords.h"
#include "model.h"

const  double rcs2 = 3.0;
const  double d_[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
       double q_[NVEL][3][3];

Site    * site;

static double eta_shear = 1.0/6.0;   /* Shear viscosity */
static double eta_bulk;              /* Bulk viscosity */
static double kT = 0.0;              /* "Isothermal temperature" */

static double rho0 = 1.0;      /* Average simulation density */
static double phi0 = 0.0;      /* Average order parameter    */


/*****************************************************************************
 *
 *  model_init
 *
 *  Set the model parameters
 *
 *****************************************************************************/

void model_init() {

  int p, i, j;

  p = RUN_get_double_parameter("viscosity", &eta_shear);
  eta_bulk = eta_shear;

  p = RUN_get_double_parameter("viscosity_bulk", &eta_bulk);
  p = RUN_get_double_parameter("temperature", &kT);
  kT = kT*rcs2; /* Without normalisation kT = cs^2 */
  p = RUN_get_double_parameter("phi0", &phi0);
  p = RUN_get_double_parameter("rho0", &rho0);

  /* Initialise q matrix */

  for (p = 0; p < NVEL; p++) {

    for (i = 0; i < 3; i++) {

      for (j = 0; j < 3; j++) {
	if (i == j) {
	  q_[p][i][j] = cv[p][i]*cv[p][j] - 1.0/rcs2;
	}
	else {
	  q_[p][i][j] = cv[p][i]*cv[p][j];
	}
      }
    }
  }

  return;
}


/***************************************************************************
 *
 *  LATT_allocate_sites
 *
 *  Allocate memory for the distributions. If MPI2 is used, then
 *  this must use the appropriate utility to accomodate LE planes.
 *
 ***************************************************************************/
 
void allocate_site(const int nsites) {

  info("Requesting %d bytes for site data\n", nsites*sizeof(Site));

#ifdef _MPI_2_
 {
   int ifail;

   ifail = MPI_Alloc_mem(nsites*sizeof(Site), MPI_INFO_NULL, &site);
   if (ifail == MPI_ERR_NO_MEM) fatal("MPI_Alloc_mem(site) failed\n");
 }
#else

  /* Use calloc. */

  site = (Site  *) calloc(nsites, sizeof(Site));
  if (site == NULL) fatal("calloc(site) failed\n");

#endif

  return;
}

/*****************************************************************************
 *
 *  set_rho
 *
 *  Project rho onto the distribution at position index, assuming zero
 *  velocity and zero stress.
 *
 *****************************************************************************/

void set_rho(const double rho, const int index) {

  int   p;

  for (p = 0; p < NVEL; p++) {
    site[index].f[p] = wv[p]*rho;
  }

  return;
}

/*****************************************************************************
 *
 *  set_rho_u_at_site
 *
 *  Project rho, u onto distribution at position index, assuming
 *  zero stress.
 *
 *****************************************************************************/

void set_rho_u_at_site(const double rho, const double u[], const int index) {

  int p;
  double udotc;

  for (p = 0; p < NVEL; p++) {
    udotc = u[X]*cv[p][X] + u[Y]*cv[p][Y] + u[Z]*cv[p][Z];
    site[index].f[p] = wv[p]*(rho + rcs2*udotc);
  }

  return;
}

/*****************************************************************************
 *
 *  set_phi
 *
 *  Sets the order parameter distribution at index address, assuming
 *  zero order parameter flux and zero stress.
 *
 *  Note that this is currently incompatible with the reprojection
 *  at the collision stage where all the phi would go into the rest
 *  distribution.
 *
 ****************************************************************************/

void set_phi(const double phi, const int index) {

  int   p;

  for (p = 0; p < NVEL; p++) {
    site[index].g[p] = wv[p]*phi;
  }

  return;
}

/*****************************************************************************
 *
 *  set_f_at_site
 *
 *****************************************************************************/

void set_f_at_site(const int index, const int p, const double fp) {

  site[index].f[p] = fp;

  return;
}

/*****************************************************************************
 *
 *  get_f_at_site
 *
 *****************************************************************************/

double get_f_at_site(const int index, const int p) {

  return site[index].f[p];
}

/*****************************************************************************
 *
 *  set_g_at_site
 *
 *****************************************************************************/

void set_g_at_site(const int index, const int p, const double gp) {

  site[index].g[p] = gp;

  return;
}

/*****************************************************************************
 *
 *  get_g_at_site
 *
 *****************************************************************************/

double get_g_at_site(const int index, const int p) {

  return site[index].g[p];
}

/*****************************************************************************
 *
 *  get_rho_at_site
 *
 *  Return the density at lattice node index.
 *
 *****************************************************************************/

double get_rho_at_site(const int index) {

  double rho;
  double * f;
  int   p;

  rho = 0.0;
  f = site[index].f;

  for (p = 0; p < NVEL; p++)
    rho += f[p];

  return rho;
}

/****************************************************************************
 *
 *  get_phi_at_site
 *
 *  Return the order parameter at lattice node index.
 *
 ****************************************************************************/

double get_phi_at_site(const int index) {

  double   phi;
  double * g;
  int     p;

  phi = 0.0;
  g = site[index].g;

  for (p = 0; p < NVEL; p++) {
    phi += g[p];
  }

  return phi;
}

/*****************************************************************************
 *
 *  get_eta_shear
 *
 *  Return the shear viscosity.
 *
 *****************************************************************************/

double get_eta_shear() {

  return eta_shear;
}

/*****************************************************************************
 *
 *  get_eta_bulk
 *
 *  Return the bulk viscosity
 *
 *****************************************************************************/

double get_eta_bulk() {

  return eta_bulk;
}

/*****************************************************************************
 *
 *  get_kT
 *
 *  Access function for the isothermal temperature.
 *
 *****************************************************************************/

double get_kT() {

  return kT;
}

/*****************************************************************************
 *
 *  get_rho0
 *
 *  Access function for the mean fluid density.
 *
 *****************************************************************************/

double get_rho0() {

  return rho0;
}

/*****************************************************************************
 *
 *  get_phi0
 *
 *  Access function for the mean order parameter.
 *
 *****************************************************************************/

double get_phi0() {

  return phi0;
}
