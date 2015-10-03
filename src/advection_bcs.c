/****************************************************************************
 *
 *  advection_bcs.c
 *
 *  Advection boundary conditions.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "wall.h"
#include "coords.h"
#include "coords_field.h"
#include "advection_s.h"
#include "advection_bcs.h"
#include "psi_gradients.h"
#include "map_s.h"

/*****************************************************************************
 *
 *  advection_bcs_no_normal_fluxes
 *
 *  Set normal fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/

__targetEntry__ void advection_bcs_no_normal_flux_lattice(int nf, advflux_t * flux, map_t * map) {


  int index;
  __targetTLP__(index,tc_nSites){

    int n;
    int nlocal[3];
    int ic, jc, kc, indexf;
    int status;
    
    double mask, maskw, maske, masky, maskz;
    
    int coords[3];
    targetCoords3D(coords,tc_Nall,index);
    
    // if not a halo site:


    if (coords[0] >= (tc_nhalo) &&
    	coords[1] >= (tc_nhalo-1) &&
    	coords[2] >= (tc_nhalo-1) &&
    	coords[0] < (tc_Nall[X]-tc_nhalo) &&
	coords[1] < (tc_Nall[Y]-tc_nhalo)  &&
	coords[2] < (tc_Nall[Z]-tc_nhalo)) {


      int index1;
      
      
      index1=targetIndex3D(coords[0]-1,coords[1],coords[2],tc_Nall);
      maskw = (map->status[index1]==MAP_FLUID);    

      index1=targetIndex3D(coords[0]+1,coords[1],coords[2],tc_Nall);
      maske = (map->status[index1]==MAP_FLUID);    
      
      index1= targetIndex3D(coords[0],coords[1]+1,coords[2],tc_Nall);
      masky = (map->status[index1]==MAP_FLUID);    
      
      index1= targetIndex3D(coords[0],coords[1],coords[2]+1,tc_Nall);
      maskz = (map->status[index1]==MAP_FLUID);    

      
      index1= targetIndex3D(coords[0],coords[1],coords[2],tc_Nall);
      mask = (map->status[index1]==MAP_FLUID);    
      
      for (n = 0;  n < nf; n++) {

	indexf = ADVADR(tc_nSites,nf,index,n);
	flux->fw[indexf] *= mask*maskw;
	flux->fe[indexf] *= mask*maske;
	flux->fy[indexf] *= mask*masky;
	flux->fz[indexf] *= mask*maskz;
      }
    }
 }


  return;
}


int advection_bcs_no_normal_flux(int nf, advflux_t * flux, map_t * map) {

  int n;
  int nlocal[3];
  int ic, jc, kc, index, indexf;
  int status;

  assert(nf > 0);
  assert(flux);
  assert(map);

  coords_nlocal(nlocal);

  int nhalo;
  nhalo = coords_nhalo();


  int Nall[3];
  Nall[X]=nlocal[X]+2*nhalo;  Nall[Y]=nlocal[Y]+2*nhalo;  Nall[Z]=nlocal[Z]+2*nhalo;

  int nSites=Nall[X]*Nall[Y]*Nall[Z];


  //copy lattice shape constants to target ahead of execution
  copyConstToTarget(&tc_nSites,&nSites, sizeof(int));
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int));
  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int));

  double* tmpptr;

#ifndef KEEPFIELDONTARGET
  map_t* t_map = map->tcopy; //target copy of field structure


  copyFromTarget(&tmpptr,&(t_map->status),sizeof(char*)); 
  copyToTarget(tmpptr,map->status,nSites*sizeof(char));


  advflux_t* t_flux = flux->tcopy; //target copy of flux structure

  copyFromTarget(&tmpptr,&(t_flux->fe),sizeof(double*));
  copyToTarget(tmpptr,flux->fe,nf*nSites*sizeof(double));

  copyFromTarget(&tmpptr,&(t_flux->fw),sizeof(double*));
  copyToTarget(tmpptr,flux->fw,nf*nSites*sizeof(double));

  copyFromTarget(&tmpptr,&(t_flux->fy),sizeof(double*));
  copyToTarget(tmpptr,flux->fy,nf*nSites*sizeof(double));

  copyFromTarget(&tmpptr,&(t_flux->fz),sizeof(double*));
  copyToTarget(tmpptr,flux->fz,nf*nSites*sizeof(double));
#endif

  advection_bcs_no_normal_flux_lattice  __targetLaunch__(nSites) (nf,flux->tcopy, map->tcopy);

#ifndef KEEPFIELDONTARGET
  copyFromTarget(&tmpptr,&(t_flux->fe),sizeof(double*));
  copyFromTarget(flux->fe,tmpptr,nf*nSites*sizeof(double));

  copyFromTarget(&tmpptr,&(t_flux->fw),sizeof(double*));
  copyFromTarget(flux->fw,tmpptr,nf*nSites*sizeof(double));

  copyFromTarget(&tmpptr,&(t_flux->fy),sizeof(double*));
  copyFromTarget(flux->fy,tmpptr,nf*nSites*sizeof(double));

  copyFromTarget(&tmpptr,&(t_flux->fz),sizeof(double*));
  copyFromTarget(flux->fz,tmpptr,nf*nSites*sizeof(double));
#endif

  return 0;
}

/*****************************************************************************
 *
 *  advective_bcs_no_flux
 *
 *  Set normal fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/

int advective_bcs_no_flux(int nf, double * fx, double * fy, double * fz,
			  map_t * map) {
  int n;
  int nlocal[3];
  int ic, jc, kc, index, indexf;
  int status;

  double mask, maskx, masky, maskz;

  assert(nf > 0);
  assert(fx);
  assert(fy);
  assert(fz);
  assert(map);

  coords_nlocal(nlocal);

  for (ic = 0; ic <= nlocal[X]; ic++) {
    for (jc = 0; jc <= nlocal[Y]; jc++) {
      for (kc = 0; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic + 1, jc, kc);
	map_status(map, index, &status);
	maskx = (status == MAP_FLUID);

	index = coords_index(ic, jc + 1, kc);
	map_status(map, index, &status);
	masky = (status == MAP_FLUID);

	index = coords_index(ic, jc, kc + 1);
	map_status(map, index, &status);
	maskz = (status == MAP_FLUID);

	index = coords_index(ic, jc, kc);
	map_status(map, index, &status);
	mask = (status == MAP_FLUID);

	for (n = 0;  n < nf; n++) {

	  coords_field_index(index, n, nf, &indexf);
	  fx[indexf] *= mask*maskx;
	  fy[indexf] *= mask*masky;
	  fz[indexf] *= mask*maskz;

	}

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  advective_bcs_no_flux_d3qx
 *
 *  Set normal fluxes at solid fluid interfaces to zero.
 *
 *****************************************************************************/

int advective_bcs_no_flux_d3qx(int nf, double ** flx, map_t * map) {

  int n;
  int nlocal[3];
  int ic, jc, kc, index0, index1;
  int status;
  int c;
  double * mask;

  assert(nf > 0);
  assert(flx);
  assert(map);

  mask = (double*) calloc(PSI_NGRAD, sizeof(double)); 

  coords_nlocal(nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = coords_index(ic, jc, kc);
	map_status(map, index0, &status);
	mask[0] = (status == MAP_FLUID);

	for (c = 1; c < PSI_NGRAD; c++) {

	  index1 = coords_index(ic + psi_gr_cv[c][X], jc + psi_gr_cv[c][Y], kc + psi_gr_cv[c][Z]);
	  map_status(map, index1, &status);
	  mask[c] = (status == MAP_FLUID);

	  for (n = 0;  n < nf; n++) {
	    flx[nf*index0 + n][c - 1] *= mask[0]*mask[c];
	  }

	}

      }
    }
  }

  return 0;
}
/*****************************************************************************
 *
 *  advection_bcs_wall
 *
 *  For the case of flat walls, we kludge the order parameter advection
 *  by borrowing the adjacent fluid value.
 *
 *  The official explanation is this may be viewed as a no gradient
 *  condition on the order parameter near the wall.
 *
 *  This allows third and fourth order (x-direction) advective fluxes
 *  to be computed at interface one cell away from wall. Fluxes at
 *  the wall will always be zero.
 *
 ****************************************************************************/

int advection_bcs_wall(field_t * fphi) {

  int ic, jc, kc, index, index1;
  int nlocal[3];
  int nf;
  double q[NQAB];

  if (wall_at_edge(X) == 0) return 0;

  assert(fphi);

  field_nf(fphi, &nf);
  coords_nlocal(nlocal);
  assert(nf <= NQAB);

  if (cart_coords(X) == 0) {
    ic = 1;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index  = coords_index(ic, jc, kc);
	index1 = coords_index(ic-1, jc, kc);

	field_scalar_array(fphi, index, q);
	field_scalar_array_set(fphi, index1, q);
      }
    }
  }

  if (cart_coords(X) == cart_size(X) - 1) {

    ic = nlocal[X];

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	index1 = coords_index(ic+1, jc, kc);

	field_scalar_array(fphi, index, q);
	field_scalar_array_set(fphi, index1, q);

      }
    }
  }

  return 0;
}
