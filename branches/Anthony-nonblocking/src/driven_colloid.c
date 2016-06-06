/*****************************************************************************
 *
 *  driven_colloid.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2014 The University of Edinburgh
 *  Contributing authors:
 *    Juho Lintuvuori (juho.lintuvuori@u-psud.fr)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "driven_colloid.h"

static double fmod_ = 0.0;
static int driven_ = 0; /*switch*/

/*****************************************************************************
 *
 *  driven_colloid_fmod_set
 *
 *****************************************************************************/

void driven_colloid_fmod_set(const double f0) {
  
  fmod_ = f0;
  if (f0 > 0.0) driven_ = 1;
  
}

/*****************************************************************************
 *
 *  driven_colloid_fmod_get
 *
 *****************************************************************************/

double driven_colloid_fmod_get(void) {

  return fmod_;
}

/*****************************************************************************
 *
 *  driven_colloid_force
 *
 *****************************************************************************/

void driven_colloid_force(const double s[3], double force[3]) {

  int ia;

  force[0] = 0.0;
  force[1] = 0.0;
  force[2] = 0.0;

  if (is_driven()){
    for (ia = 0; ia < 3; ia++){
      force[ia] = fmod_*s[ia];
    }
  }
  return;
}

/*****************************************************************************
 *
 *  driven_colloid_total_force
 *
 *****************************************************************************/

void driven_colloid_total_force(colloids_info_t * cinfo, double ftotal[3]) {

  int ia;
  double flocal[3],f[3];
  colloid_t * pc = NULL;

  assert(cinfo);

  for (ia = 0; ia < 3; ia++) {
    ftotal[ia] = 0.0;
    flocal[ia] = 0.0;
  }

  colloids_info_local_head(cinfo, &pc);

  for ( ; pc; pc = pc->nextlocal) { 

    driven_colloid_force(pc->s.s, f);
	  
    for (ia = 0; ia < 3; ia++){
      flocal[ia] += f[ia];
    }
  }
  
  MPI_Allreduce(flocal, ftotal, 3, MPI_DOUBLE, MPI_SUM, cart_comm());

  return;
}

/*****************************************************************************
 *
 *  is_driven
 *
 *****************************************************************************/

int is_driven(void) {
  
  return driven_;
}

