/*****************************************************************************
 *
 *  phi_force_stress.c
 *  
 *  Wrapper functions for stress computation.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "free_energy.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "util.h"
#include "math.h"
#include "blue_phase.h"
#include "timer.h"

#include "pth_s.h"

/*****************************************************************************
 *
 *  pth_create
 *
 *****************************************************************************/

__host__ int pth_create(int method, pth_t ** pobj) {

  int ndevice;
  double * tmp;
  pth_t * obj = NULL;

  assert(pobj);

  obj = (pth_t *) calloc(1, sizeof(pth_t));
  if (obj == NULL) fatal("calloc(pth_t) failed\n");

  obj->method = method;
  obj->nsites = coords_nsites();

  /* If memory required */

  if (method == PTH_METHOD_DIVERGENCE) {
    obj->str = (double *) calloc(NVECTOR*NVECTOR*obj->nsites, sizeof(double));
    if (obj->str == NULL) fatal("calloc(pth->str) failed\n");
  }

  /* Allocate target memory, or alias */

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {

    targetCalloc((void **) &obj->target, sizeof(pth_t));
    copyToTarget(&obj->target->nsites, &obj->nsites, sizeof(int));

    if (method == PTH_METHOD_DIVERGENCE) {
      targetCalloc((void **) &tmp, NVECTOR*NVECTOR*obj->nsites*sizeof(double));
      copyToTarget(&obj->target->str, &tmp, sizeof(double *));
    }
  }

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  pth_free
 *
 *****************************************************************************/

__host__ int pth_free(pth_t * pth) {

  int ndevice;
  double * tmp = NULL;

  assert(pth);

  targetGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    copyFromTarget(&tmp, &pth->target->str, sizeof(double *));
    if (tmp) targetFree(tmp);
    targetFree(pth->target);
  }

  if (pth->str) free(pth->str);
  free(pth);

  return 0;
}

/*****************************************************************************
 *
 *  pth_memcpy
 *
 *****************************************************************************/

__host__ int pth_memcpy(pth_t * pth, int flag) {

  int ndevice;

  assert(pth);

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Ensure we alias */
    assert(pth->target == pth);
  }
  else {
    assert(0); /* Copy */
  }

  return 0;
}

/*****************************************************************************
 *
 *  phi_force_stress_compute
 *
 *  Compute the stress everywhere and store.
 *
 *****************************************************************************/

__host__
int pth_stress_compute(pth_t * pth, fe_t * fe) {

  int ic, jc, kc, index;
  int nlocal[3];
  int nextra = 1;

  double s[3][3];

  assert(pth);
  assert(fe);

  coords_nlocal(nlocal);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(ic, jc, kc);
	fe->func->stress(fe, index, s);
	pth_stress_set(pth, index, s);
      }
    }
  }

  return 0;
}

#ifdef OLD_SHIT
__targetEntry__
void chemical_stress_lattice(pth_t * pth, field_t * q, field_grad_t * qgrad,
			     void * pcon,
			     void (* chemical_stress)(const int index, double s[3][3]),int isBPCS){ 

  int baseIndex;

  __targetTLP__(baseIndex, tc_nSites) {


#if VVL == 1    
    /*restrict operation to the interior lattice sites*/ 

    int coords[3];
    targetCoords3D(coords,tc_Nall,baseIndex);

    /*  if not a halo site:*/
    if (coords[0] >= (tc_nhalo-tc_nextra) &&
    	coords[1] >= (tc_nhalo-tc_nextra) &&
    	coords[2] >= (tc_nhalo-tc_nextra) &&
    	coords[0] < tc_Nall[X]-(tc_nhalo-tc_nextra) &&
    	coords[1] < tc_Nall[Y]-(tc_nhalo-tc_nextra)  &&
    	coords[2] < tc_Nall[Z]-(tc_nhalo-tc_nextra) )
#endif

      { 
      
      if (isBPCS){
	/* we are using blue_phase_chemical_stress which is ported to targetDP
	 * for the time being we are explicitly calling
	 * blue_phase_chemical_stress
	 * ultimitely this will be generic when the other options are
	 * ported to targetDP */
	 int calledFromPhiForceStress=1;
	 blue_phase_chemical_stress_dev_vec(baseIndex, q, qgrad, pth->str,
					    pcon, 
					    calledFromPhiForceStress);
      }
      else{

	double pth_local[3][3];

#ifndef __NVCC__

#if VVL > 1
	fatal("Vectorisation not yet supported for this chemical stress");
#endif

	/* only blue_phase_chemical_stress support for CUDA. */
        /* TO DO: support vectorisation for these routines */  
	chemical_stress(baseIndex, pth_local);
	phi_force_stress_set(pth, baseIndex, pth_local); 
#endif
      }
      
    }
  }

return;
}

/*****************************************************************************
 *
 *  phi_force_stress_compute
 *
 *****************************************************************************/

__host__
int phi_force_stress_compute(pth_t * pth, field_t * q, field_grad_t * qgrad) {

  int nlocal[3];
  int nextra = 1;

  coords_nlocal(nlocal);
  assert(coords_nhalo() >= 2);

  int nhalo = coords_nhalo();
  int Nall[3];
  Nall[X]=nlocal[X]+2*nhalo;  Nall[Y]=nlocal[Y]+2*nhalo;  Nall[Z]=nlocal[Z]+2*nhalo;
  int nSites=Nall[X]*Nall[Y]*Nall[Z];

  void (* chemical_stress)(const int index, double s[3][3]);

  chemical_stress = fe_chemical_stress_function();

  
  /* initialise kernel constants on both host and target */
  blue_phase_set_kernel_constants();

  void* pcon=NULL;
  blue_phase_target_constant_ptr(&pcon);

  /* isBPCS is 1 if we are using  blue_phase_chemical_stress 
   * (which is ported to targetDP), 0 otherwise*/

  int isBPCS=((void*)chemical_stress)==((void*) blue_phase_chemical_stress);

#ifdef __NVCC__
    if (!isBPCS) fatal("only Blue Phase chemical stress is currently supported for CUDA");
#endif


  copyConstToTarget(&tc_nSites,&nSites, sizeof(int));
  copyConstToTarget(&tc_nextra,&nextra, sizeof(int));
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int));
  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int));

  TIMER_start(TIMER_CHEMICAL_STRESS_KERNEL);

  chemical_stress_lattice __targetLaunch__(nSites) (pth->target, q->target, qgrad->tcopy, pcon, chemical_stress, isBPCS);
  targetSynchronize();

  TIMER_stop(TIMER_CHEMICAL_STRESS_KERNEL);
 
  return 0;
}
#endif

/*****************************************************************************
 *
 *  phi_force_stress_set
 *
 *****************************************************************************/

__host__  __device__
void pth_stress_set(pth_t * pth, int index, double p[3][3]) {

  int ia, ib;

  assert(pth);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      pth->str[addr_rank2(pth->nsites,3,3,index,ia,ib)] = p[ia][ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_stress
 *
 *****************************************************************************/

__host__  __device__
void pth_stress(pth_t * pth, int index, double p[3][3]) {

  int ia, ib;

  assert(pth);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      p[ia][ib] = pth->str[addr_rank2(pth->nsites,3,3,index,ia,ib)];
    }
  }

  return;
}
