/*****************************************************************************
 *
 *  test_fft_sor.c
  *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Ruairi Short (Rshort@sms.ed.ac.uk)
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
#include "psi_s.h"
#include "psi_sor.h"
#include "timer.h"
#include "psi_stats.h"
#include "psi_colloid.h"

#include "psi_fft.h"

#include "util.h"
#include "psi_stats.h"

#include "runtime.h"
#include "coords_rt.h"

#define REF_PERMEATIVITY 1.0

int main(int argc, char ** argv) {

  char inputfile[FILENAME_MAX] = "input";
  int i,j,k;
  int index = 0;
  int nlocal[3] = {0, 0, 0};
  int global_coord[3] = {0, 0, 0};
  int global_coord_save[3] = {0, 0, 0};
  int iter = 1;
  double pi = 4.0*atan(1.0); 
  double max;

  MPI_Init(&argc, &argv);

  if (argc > 1) sprintf(inputfile, "%s", argv[1]);

  
  pe_init();
 
  RUN_read_input_file(inputfile);

  coords_run_time();

//  coords_init();
  decomp_init();

  psi_t *psi_sor = NULL;
  psi_t *psi_fft = NULL;
  
  psi_create(2, &psi_sor);
  psi_create(2, &psi_fft);
  assert(psi_sor);
  assert(psi_fft);

  psi_valency_set(psi_sor, 0, +1.0);
  psi_valency_set(psi_sor, 1, -1.0);
  psi_epsilon_set(psi_sor, REF_PERMEATIVITY);

  psi_valency_set(psi_fft, 0, +1.0);
  psi_valency_set(psi_fft, 1, -1.0);
  psi_epsilon_set(psi_fft, REF_PERMEATIVITY);

  coords_nlocal(nlocal);
  coords_nlocal_offset(global_coord);
  coords_nlocal_offset(global_coord_save);

  /*set up psi_sor and psi_fft to be sin functions, these will be charge neutral as they are over one period*/ 
  for(i=1; i<=nlocal[X]; i++) {
    for(j=1; j<=nlocal[Y]; j++) {
      for(k=1; k<=nlocal[Z]; k++) {
          index = coords_index(i, j, k);
          psi_psi_set(psi_fft, index, 0.0);
          psi_psi_set(psi_sor, index, 0.0);

          psi_rho_set(psi_fft, index, 1, sin(2*pi*global_coord[Z]/N_total(Z))* sin(2*pi*global_coord[Y]/N_total(Y))* sin(2*pi*global_coord[X]/N_total(X)) );
          psi_rho_set(psi_sor, index, 1, sin(2*pi*global_coord[Z]/N_total(Z))* sin(2*pi*global_coord[Y]/N_total(Y))* sin(2*pi*global_coord[X]/N_total(X)) );
          psi_rho_set(psi_sor, index, 0, 0.0);
          psi_rho_set(psi_fft, index, 0, 0.0);
        global_coord[Z] ++;
      }
      global_coord[Z] = global_coord_save[Z];
      global_coord[Y] ++;
    }
    global_coord[Y] = global_coord_save[Y];
    global_coord[X] ++;
  } 


  for(i=0; i<iter; i++) { 
   /*use psi_sor_poisson to solve*/
    if(pe_rank() == 0) { printf("Solving with SOR\n"); }
      psi_sor_poisson(psi_sor);
  }


  for(i=0; i<iter; i++) { 
    /*use psi_fft_poisson to solve*/
    if(pe_rank() == 0) { printf("Solving with FFT\n"); }
    psi_fft_poisson(psi_fft);
  }  


/*  for(i=1; i<=nlocal[X]; i++) {
    for(j=1; j<=nlocal[Y]; j++) {
      for(k=1; k<=nlocal[Z]; k++) {
        printf("%d %d %f %f\n", i, j, psi_fft->psi[coords_index(i,j,k)], psi_sor->psi[coords_index(i,j,k)]);
      }
    }
  }*/

  /*check results are acceptably similar */
  for(i=1; i<=nlocal[X]; i++) {
    for(j=1; j<=nlocal[Y]; j++) {
      for(k=1; k<=nlocal[Z]; k++) {
        if(fabs(psi_sor->psi[coords_index(i,j,k)]) > 1e-8) {
          if(fabs( (fabs(psi_sor->psi[coords_index(i,j,k)] - psi_fft->psi[coords_index(i,j,k)])) / (psi_sor->psi[coords_index(i,j,k)]) >= 0.01 )) { 
            printf("fft: %e, sor: %e diff %f\n", psi_fft->psi[coords_index(i,j,k)], psi_sor->psi[coords_index(i,j,k)], fabs( (fabs(psi_sor->psi[coords_index(i,j,k)] - psi_fft->psi[coords_index(i,j,k)])) / (psi_sor->psi[coords_index(i,j,k)]) ) );
          }

          assert( fabs( (fabs(psi_sor->psi[coords_index(i,j,k)] - psi_fft->psi[coords_index(i,j,k)])) / (psi_sor->psi[coords_index(i,j,k)]) ) < 0.01);
        }
        else {
//          printf("%f\n", psi_fft->psi[coords_index(i,j,k)]);
          assert( fabs(psi_fft->psi[coords_index(i,j,k)]) < 5e-8);
        }
      }
    }
  }




  psi_free(psi_sor);
  psi_free(psi_fft);

  decomp_finish();
  coords_finish();
  pe_finalise();
  MPI_Finalize();

  return 0;

}
