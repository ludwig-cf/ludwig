/*****************************************************************************
*
*  test_nernst_planck.c
*
*  Unit test for electrokinetic quantities.
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

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "site_map.h"
#include "psi.h"
#include "psi_s.h"
#include "psi_sor.h"
#include "psi_stats.h"
#include "nernst_planck.h"

static int do_test_gouy_chapman(void);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  MPI_Init(&argc, &argv);
  pe_init();

  do_test_gouy_chapman();

  pe_finalise();
  MPI_Finalize();

  return 0;
}
/*****************************************************************************
 *
 *  do_test_gouy_chapman
 *
 *  Set rho(z = 1)  = + (1/2NxNy)
 *      rho(z = Lz) = + (1/2NxNy)
 *      rho         = - 1/(NxNy*(Nz-2)) + electrolyte
 *
 *  This is a test of the Gouy-Chapman theory.
 *
 *****************************************************************************/

static int do_test_gouy_chapman(void) {

  int ic, jc, kc, index, in;
  int nlocal[3], noff[3];
  int tstep;
  double rho_w, rho_i, rho_el, ios;
  double field[3], lb[1], ld[1], cont[1];
  double tol_abs = 0.01*FLT_EPSILON;
  double tol_rel = 1.00*FLT_EPSILON;
  double diffusivity[2] = {1.e-2, 1.e-2};
  double eunit = 1.;
  double epsilon = 3.3e3;
  double beta = 3.0e4;
  FILE * out;
  int n[3]={4,4,64};
  int tmax = 101;
  char filename[30];

  coords_nhalo_set(1);
  coords_ntotal_set(n);
  coords_init();
  coords_nlocal(nlocal);

  site_map_init();

  psi_create(2, &psi_);

  psi_valency_set(psi_, 0, +1);
  psi_valency_set(psi_, 1, -1);
  psi_diffusivity_set(psi_, 0, diffusivity[0]);
  psi_diffusivity_set(psi_, 1, diffusivity[1]);
  psi_unit_charge_set(psi_, eunit);
  psi_epsilon_set(psi_, epsilon);
  psi_beta_set(psi_, beta);

  /* wall charge density */
  rho_w = 1.e+0 / (2.0*L(X)*L(Y));
  rho_el = 1.e-3;

  /* counter charge density */
  rho_i = rho_w * (2.0*L(X)*L(Y)) / (L(X)*L(Y)*(L(Z) - 2.0));

  /* apply counter charges & electrolyte */
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	psi_psi_set(psi_, index, 0.0);
	psi_rho_set(psi_, index, 0, rho_el);
	psi_rho_set(psi_, index, 1, rho_el + rho_i);

      }
    }
  }

  /* apply wall charges */
  if (cart_coords(Z) == 0) {
    kc = 1;
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {

	index = coords_index(ic, jc, kc);

	site_map_set_status(ic,jc,kc,BOUNDARY);

	psi_rho_set(psi_, index, 0, rho_w);
	psi_rho_set(psi_, index, 1, 0.0);

      }
    }
  }

  if (cart_coords(Z) == cart_size(Z) - 1) {
    kc = nlocal[Z];
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {

	index = coords_index(ic, jc, kc);

	site_map_set_status(ic,jc,kc,BOUNDARY);

	psi_rho_set(psi_, index, 0, rho_w);
	psi_rho_set(psi_, index, 1, 0.0);

      }
    }
  }

  site_map_halo();

  for(tstep=0; tstep<tmax; tstep++){

    psi_halo_psi(psi_);
    psi_sor_poisson(psi_, tol_abs, tol_rel);
    psi_halo_rho(psi_);
    nernst_planck_driver(psi_);

    if (tstep%1000==0){

      printf("%d\n", tstep);
      psi_stats_info(psi_);
      sprintf(filename,"np_test-%d.dat",tstep);
      out=fopen(filename,"w");

      ic=2;
      jc=2;

      for(kc=1; kc<=nlocal[Z]; kc++){

	index = coords_index(ic, jc, kc);

	psi_psi(psi_,index,cont);
	field[0] = cont[0];
	psi_rho(psi_,index,0,cont);
	field[1] = cont[0];
	psi_rho(psi_,index,1,cont);
	field[2] = cont[0];

	fprintf(out,"%d %le %le %le\n", kc, field[0], field[1], field[2]);

      }

      fclose(out);

    }
  }

  coords_nlocal_offset(noff);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
 
	if(noff[0]+ic == 2 && noff[1]+jc ==2 && noff[2]+kc == 0.5*n[2]){
	  for (in = 0; in < psi_->nk; in++) {
	    ios += 0.5*psi_->valency[in]*psi_->valency[in]*psi_->rho[psi_->nk*index + in];
	  }
	  psi_debye_length(psi_, ios, ld);
	}

      }
    }
  }

  if (cart_rank() == 0){
    psi_bjerrum_length(psi_,lb);
    printf("Bjerrum length is %le\n",lb[0]);
  }

  if (cart_rank() == 0){
    printf("Debye length is %le\n",ld[0]);
  }

  psi_free(psi_);
  coords_finish();
  site_map_finish();

}

