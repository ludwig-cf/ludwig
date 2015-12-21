
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "memory.h"

int main(int argc, char ** argv) {

  int nx = 2;
  int ny = 5;
  int nz = 4;
  int nsites = nx*ny*nz;
  int na = 3;

  int index, ic, jc, kc;
  int ia, iv;
  int ithread, itrip;
  int baseindex;

  int * irefindex = NULL;

  irefindex = (int *) calloc(nsites*na, sizeof(int));
  assert(irefindex);

  /* Loop 1 */

  itrip = 0;

  for (ic = 0; ic < nx; ic++) {
    for (jc = 0; jc < ny; jc++) {
      for (kc = 0; kc < nz; kc++) {
	index = ny*nz*ic + nz*jc + kc;
	for (ia = 0; ia < 3; ia++) {
	  printf("Loop1 trip, addr: %8d %8d\n", itrip++,
		 model_addr_rank1(nsites, na, index, ia));
	  irefindex[model_addr_rank1(nsites, na, index, ia)] = itrip;
 	}
      }
    }
  }

  /* Loop 2 */

  itrip = 0;

  for (ithread = 0; ithread < nsites; ithread++) {
    index = ithread;
    for (ia = 0; ia < 3; ia++) {
      printf("Loop2 trip, addr: %8d %8d\n", itrip++,
	     model_addr_rank1(nsites, na, index, ia));
      assert(irefindex[model_addr_rank1(nsites, na, index, ia)] == itrip);
     }
  }

  /* Loop 3 */

  itrip = 0;

  for (ithread = 0; ithread < nsites; ithread += nsimd) {
    baseindex = ithread;
    for (ia = 0; ia < 3; ia++) {
      for (iv = 0; iv < nsimd; iv++) {
	printf("Loop3 trip, addr: %8d %8d\n", itrip++,
	       model_addrv_rank1(nsites, na, baseindex, ia, iv));
      }    
    }
  }

  return 0;
}
