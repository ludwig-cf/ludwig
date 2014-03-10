#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "colloid.h"
#include "colloids.h"
#include "driven_colloid.h"

static double fmod_ = 0.0;
static int driven_ = 0; /*switch*/

/*****************************************************************************
 *
 *  driven_colloid_fmod_set
 *
 *****************************************************************************/

void driven_colloid_fmod_set(const double f0){
  
  fmod_ = f0;
  if(f0 > 0.0)driven_ = 1;
  
}

/*****************************************************************************
 *
 *  driven_colloid_fmod_get
 *
 *****************************************************************************/

double driven_colloid_fmod_get(void){
  double f0;
  
  f0 = fmod_;
  return(f0);
}

/*****************************************************************************
 *
 *  driven_colloid_force
 *
 *****************************************************************************/

void driven_colloid_force(const double s[3], double force[3]){
  int ia;

  for (ia = 0; ia < 3; ia++){
    force[ia] = fmod_*s[ia];
  }
}

/*****************************************************************************
 *
 *  driven_colloid_total_force
 *
 *****************************************************************************/

void driven_colloid_total_force(double ftotal[3]){

  int ic, jc, kc, ia;
  double flocal[3],f[3];
  colloid_t * pc;

  for (ia = 0; ia < 3; ia++){
    ftotal[ia] = 0.0;
    flocal[ia] = 0.0;
  }
  
  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	pc = colloids_cell_list(ic, jc, kc);

	while (pc) {
	  driven_colloid_force(pc->s.s, f);
	  
	  for (ia = 0; ia < 3; ia++){
	    flocal[ia] += f[ia];
	  }
	  pc = pc->next;
	}
      }
    }
  }
  
  MPI_Allreduce(flocal, ftotal, 3, MPI_DOUBLE, MPI_SUM, cart_comm());
}

/*****************************************************************************
 *
 *  is_driven
 *
 *****************************************************************************/

int is_driven(void){
  
  return(driven_);
}
