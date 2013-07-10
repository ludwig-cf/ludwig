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

#include "psi_fft.h"

#include "util.h"
#include "psi_stats.h"

#define REF_PERMEATIVITY 1.0

int main() {

  int i,j,k;
  int index = 0;
  int nlocal[3] = {0, 0, 0};
  int global_coord[3] = {0, 0, 0};
  int global_coord_save[3] = {0, 0, 0};
  int iter = 1;
  double pi = 4.0*atan(1.0); 
  double max;

  MPI_Init(NULL, NULL);
  
  pe_init();
  coords_init();

  TIMER_init();

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

  coords_info();

  coords_nlocal(nlocal);
  coords_nlocal_offset(global_coord);
  coords_nlocal_offset(global_coord_save);

  /*set up psi_sor and psi_fft to be cos functions*/ 
  for(i=1; i<=nlocal[X]; i++) {
    for(j=1; j<=nlocal[Y]; j++) {
      for(k=1; k<=nlocal[Z]; k++) {
          index = coords_index(i, j, k);
          psi_psi_set(psi_fft, index, 0.0);
          psi_psi_set(psi_sor, index, 0.0);
          psi_rho_set(psi_fft, index, 0, 0.0);
          psi_rho_set(psi_sor, index, 0, 0.0);

/*         if(global_coord[0] == N_total(0)/2 && global_coord[1] == N_total(1)/2 && global_coord[2] == N_total(2)/2) {
            psi_rho_set(psi_fft, index, 1, -10.0);
            psi_rho_set(psi_sor, index, 1, -10.0);
            printf("success\n");
          }
        else {*/
          psi_rho_set(psi_fft, index, 1, cos(2*pi*global_coord[Z]/N_total(Z))*cos(2*pi*global_coord[Y]/N_total(Y))*cos(2*pi*global_coord[X]/N_total(X)) );
          psi_rho_set(psi_sor, index, 1, cos(2*pi*global_coord[Z]/N_total(Z))*cos(2*pi*global_coord[Y]/N_total(Y))*cos(2*pi*global_coord[X]/N_total(X)) );
//        }
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
   TIMER_start(TIMER_PSI_SOR_UPDATE);
  if(pe_rank() == 0) { printf("Solving with SOR\n"); }
    psi_sor_poisson(psi_sor);
   TIMER_stop(TIMER_PSI_SOR_UPDATE);
}


for(i=0; i<iter; i++) { 
  /*use psi_fft_poisson to solve*/
   TIMER_start(TIMER_PSI_FFT_UPDATE);
  if(pe_rank() == 0) { printf("Solving with FFT\n"); }
  psi_fft_poisson(psi_fft);
   TIMER_stop(TIMER_PSI_FFT_UPDATE);
}  

  TIMER_statistics();

  max = 0.0;
  if(pe_rank() == 0) { printf("Checking results\n"); }
  /*check results are acceptably similar */
  for(i=1; i<=nlocal[X]; i++) {
    for(j=1; j<=nlocal[Y]; j++) {
      for(k=1; k<=nlocal[Z]; k++) {
/*        if(pe_rank() == 0) {
          printf("%f %f\n", psi_fft->psi[coords_index(i,j,k)], psi_sor->psi[coords_index(i,j,k)]);
          }
*/
        if(fabs(psi_fft->psi[coords_index(i,j,k)] - psi_sor->psi[coords_index(i,j,k)]) > max) {
          max = fabs(psi_fft->psi[coords_index(i,j,k)] - psi_sor->psi[coords_index(i,j,k)]);
        }
//        assert(psi_sor->psi[coords_index(i,j,k)] - psi_fft->psi[coords_index(i,j,k)] < 1e-5);
      }
    }
  }

  printf("rank: %d, max: %f\n", pe_rank(), max);


  psi_free(psi_sor);
  psi_free(psi_fft);

  psi_fft_clean();
  coords_finish();
  pe_finalise();
  MPI_Finalize();

  return 0;

}
