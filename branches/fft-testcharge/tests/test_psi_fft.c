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

#include "psi_fft.h"

#include "util.h"
#include "psi_stats.h"

int main() {

  int i,j,k;
  int nlocal[3] = {0, 0, 0};

  MPI_Init(NULL, NULL);
  
  pe_init();
  coords_init();

  psi_t *psi_sor = NULL;
  psi_t *psi_fft = NULL;
  
  psi_create(2, &psi_sor);
  psi_create(2, &psi_fft);
  assert(psi_sor);
  assert(psi_fft);

  coords_nlocal(nlocal);

  /*set up psi_sor and psi_fft to be all 0s except for the center element*/ 
  for(i=0; i<nlocal[0]; i++) {
    for(j=0; j<nlocal[1]; j++) {
      for(k=0; k<nlocal[2]; k++) {
        if(cart_coords(X) == cart_size(X)/2 && cart_coords(Y) == cart_size(Y)/2 && cart_coords(Z) == cart_size(Z)/2
          && i == nlocal[0]/2 && j == nlocal[1]/2 && k == nlocal[2]/2)  {
            psi_sor->psi[coords_index(i,j,k)] = psi_fft->psi[coords_index(i,j,k)] = 1;
          }
        else {
          psi_sor->psi[coords_index(i,j,k)] = psi_fft->psi[coords_index(i,j,k)] = 0;
        }
      }
    }
  } 

  /*use psi_sor_poisson to solve*/
  if(pe_rank() == 0) { printf("Solving with SOR\n"); }
  psi_sor_poisson(psi_sor);

  /*use psi_fft_poisson to solve*/
  if(pe_rank() == 0) { printf("Solving with FFT\n"); }
  psi_fft_poisson(psi_fft);

  if(pe_rank() == 0) { printf("Checking results\n"); }
  /*check results are acceptably similar */
  for(i=0; i<nlocal[0]; i++) {
    for(j=0; j<nlocal[1]; j++) {
      for(k=0; k<nlocal[2]; k++) {
          assert(psi_sor->psi[coords_index(i,j,k)] - psi_fft->psi[coords_index(i,j,k)] < 1e-5);
      }
    }
  }


  psi_free(psi_sor);
  psi_free(psi_fft);

  coords_finish();
  pe_finalise();
  MPI_Finalize();

  return 0;

}
