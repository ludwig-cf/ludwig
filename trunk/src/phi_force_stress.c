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
 *  (c) 2012 The University of Edinburgh
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
#include "phi_force_stress.h"
#include "util.h"
#include "math.h"
#include "blue_phase.h"


static double * pth_;
static double * t_pth_;

/*****************************************************************************
 *
 *  phi_force_stress_compute
 *
 *  Compute the stress everywhere and store.
 *
 *****************************************************************************/

extern __targetConst__ int tc_nSites; 
extern __targetConst__ int tc_nhalo;
extern __targetConst__ int tc_nextra;  
extern __targetConst__ int tc_Nall[3]; 

__targetEntry__ void chemical_stress_lattice(double pth_local[3][3], field_t* t_q, field_grad_t* t_q_grad, double* t_pth, void* pcon, void (* chemical_stress)(const int index, double s[3][3])){ 

  int index;

__targetTLP__(index,tc_nSites){
    
    int coords[3];
    targetCoords3D(coords,tc_Nall,index);
    
    // if not a halo site:
    if (coords[0] >= (tc_nhalo-tc_nextra) && 
	coords[1] >= (tc_nhalo-tc_nextra) && 
	coords[2] >= (tc_nhalo-tc_nextra) &&
	coords[0] < tc_Nall[X]-(tc_nhalo-tc_nextra) &&  
	coords[1] < tc_Nall[Y]-(tc_nhalo-tc_nextra)  &&  
	coords[2] < tc_Nall[Z]-(tc_nhalo-tc_nextra) ){ 
      

      if (t_q){ //we are using blue_phase_chemical_stress which is ported to targetDP
	//for the time being we are explicitly calling blue_phase_chemical_stress
	//ultimitely this will be generic when the other options are ported to targetDP
	int calledFromPhiForceStress=1;
	blue_phase_chemical_stress(index, t_q, t_q_grad, t_pth, pcon, 
				   calledFromPhiForceStress);
      }
      else{

#ifndef CUDA //only blue_phase_chemical_stress support for CUDA. This is trapped earlier.
	chemical_stress(index, pth_local);
	phi_force_stress_set(index, pth_local); 
#endif

      }
      
    }
  }

return;
}

__targetHost__ void phi_force_stress_compute(field_t * q, field_grad_t* q_grad) {

  int nlocal[3];
  int nextra = 1;
  coords_nlocal(nlocal);
  assert(coords_nhalo() >= 2);

  int nhalo = coords_nhalo();
  int Nall[3];
  Nall[X]=nlocal[X]+2*nhalo;  Nall[Y]=nlocal[Y]+2*nhalo;  Nall[Z]=nlocal[Z]+2*nhalo;
  int nSites=Nall[X]*Nall[Y]*Nall[Z];

  double pth_local[3][3];
  void (* chemical_stress)(const int index, double s[3][3]);


  chemical_stress = fe_chemical_stress_function();

  //start targetDP
  
  // initialise kernel constants on both host and target
  blue_phase_set_kernel_constants();

  // get a pointer to target copy of stucture containing kernel constants
  void* pcon=NULL;
  blue_phase_target_constant_ptr(&pcon);

  field_t* t_q = q->tcopy; //target copy of tensor order parameter field structure
  field_grad_t* t_q_grad = q_grad->tcopy; //target copy of grad field structure

  if (q){ //we are using blue_phase_chemical_stress which is ported to targetDP

    double* tmpptr;
    
    //populate target copies from host 
    copyFromTarget(&tmpptr,&(t_q->data),sizeof(double*)); 
    copyToTarget(tmpptr,q->data,q->nf*nSites*sizeof(double));
    
    copyFromTarget(&tmpptr,&(t_q_grad->grad),sizeof(double*)); 
    copyToTarget(tmpptr,q_grad->grad,q_grad->nf*NVECTOR*nSites*sizeof(double));
    
    copyFromTarget(&tmpptr,&(t_q_grad->delsq),sizeof(double*)); 
    copyToTarget(tmpptr,q_grad->delsq,q_grad->nf*nSites*sizeof(double));
    
  }
  else{
#ifdef CUDA
    fatal("Error: only blue_phase_chemical_stress is currently supported for CUDA\n");
#endif
  }
  


  //copy lattice shape constants to target ahead of execution
  copyConstToTarget(&tc_nSites,&nSites, sizeof(int));
  copyConstToTarget(&tc_nextra,&nextra, sizeof(int));
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int));
  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int));
  

  //execute lattice-based operation on target
  chemical_stress_lattice __targetLaunch__(nSites) (pth_local, t_q, t_q_grad, t_pth_, pcon, chemical_stress);
  
  if (q){ //we are using blue_phase_chemical_stress which is ported to targetDP
    //copy result from target back to host
    copyFromTarget(pth_,t_pth_,3*3*nSites*sizeof(double));      
  }
  
  //end targetDP
  


  return;
}

/*****************************************************************************
 *
 *  phi_force_stress_set
 *
 *****************************************************************************/

__targetHost__  void phi_force_stress_set(const int index, double p[3][3]) {

  int ia, ib, n;

  assert(pth_);

  n = 0;
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      pth_[9*index + n++] = p[ia][ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_stress
 *
 *****************************************************************************/

__targetHost__  void phi_force_stress(const int index, double p[3][3]) {

  int ia, ib, n;

  assert(pth_);

  n = 0;
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      p[ia][ib] = pth_[9*index + n++];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_stress_allocate
 *
 *****************************************************************************/

__targetHost__  void phi_force_stress_allocate() {

  int n;

  assert(coords_nhalo() >= 2);

  n = coords_nsites();

  pth_ = (double *) malloc(9*n*sizeof(double));
  if (pth_ == NULL) fatal("malloc(pth_) failed\n");

  targetMalloc((void**) &t_pth_,9*n*sizeof(double));


  return;
}

/*****************************************************************************
 *
 *  phi_force_stress_free
 *
 *****************************************************************************/

__targetHost__  void phi_force_stress_free() {

  free(pth_);
  targetFree(t_pth_);

  return;
}
