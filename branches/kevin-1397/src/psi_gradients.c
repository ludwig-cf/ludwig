/****************************************************************************
 *
 *  psi_gradients.c
 *
 *  Finite difference stencils used in the 
 *  electrokinetic routines.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *  (c) 2014 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include "coords.h"
#include "psi_s.h"
#include "psi_gradients.h"
#include "fe_electro_symmetric.h"

#ifdef NP_D3Q6
const int psi_gr_cv[PSI_NGRAD][3] = {{ 0,  0,  0},
		{-1,  0,  0}, { 0, -1,  0}, { 0,  0, -1},
		{ 1,  0,  0}, { 0,  1,  0}, { 0,  0,  1}};

#define w0 (0.0)
#define w1  (1.0/6.0)

const double psi_gr_wv[PSI_NGRAD] = {w0,
                         	     w1, w1, w1, 
			             w1, w1, w1};

const double psi_gr_rnorm[PSI_NGRAD] = {0.0,
		                        1.0, 1.0, 1.0,
                	                1.0, 1.0, 1.0};

const double psi_gr_rcs2 = 3.0;
#endif

#ifdef NP_D3Q18
const int psi_gr_cv[PSI_NGRAD][3] = {{ 0,  0,  0},
		{ 1,  1,  0}, { 1,  0,  1}, { 1,  0,  0},
		{ 1,  0, -1}, { 1, -1,  0}, { 0,  1,  1},
		{ 0,  1,  0}, { 0,  1, -1}, { 0,  0,  1},
		{ 0,  0, -1}, { 0, -1,  1}, { 0, -1,  0},
		{ 0, -1, -1}, {-1,  1,  0}, {-1,  0,  1},
		{-1,  0,  0}, {-1,  0, -1}, {-1, -1,  0}};


#define w0 (12.0/36.0)
#define w1  (2.0/36.0)
#define w2  (1.0/36.0)

#define sqrt2 1.4142135623730951 

const double psi_gr_wv[PSI_NGRAD] = {w0,
                        w2, w2, w1, w2, w2, w2, 
			w1, w2, w1, w1, w2, w1, 
			w2, w2, w2, w1, w2, w2};

const double psi_gr_rnorm[PSI_NGRAD] = {0.0,
		1.0/sqrt2, 1.0/sqrt2, 1.0, 1.0/sqrt2, 1.0/sqrt2, 1.0/sqrt2,
		1.0, 1.0/sqrt2, 1.0, 1.0, 1.0/sqrt2, 1.0, 
		1.0/sqrt2, 1.0/sqrt2, 1.0/sqrt2, 1.0, 1.0/sqrt2, 1.0/sqrt2};

const double psi_gr_rcs2 = 3.0;
#endif

#ifdef NP_D3Q26
const int psi_gr_cv[PSI_NGRAD][3] = {{ 0, 0, 0}, 
		{-1,-1,-1}, {-1,-1, 0}, {-1,-1, 1}, 
		{-1, 0,-1}, {-1, 0, 0}, {-1, 0, 1}, 
		{-1, 1,-1}, {-1, 1, 0}, {-1, 1, 1}, 
		{ 0,-1,-1}, { 0,-1, 0}, { 0,-1, 1}, 
		{ 0, 0,-1},             { 0, 0, 1}, 
		{ 0, 1,-1}, { 0, 1, 0}, { 0, 1, 1}, 
		{ 1,-1,-1}, { 1,-1, 0}, { 1,-1, 1}, 
		{ 1, 0,-1}, { 1, 0, 0}, { 1, 0, 1}, 
		{ 1, 1,-1}, { 1, 1, 0}, { 1, 1, 1}};


#define w0 (8.0/27.0)
#define w1 (2.0/27.0)
#define w2 (1.0/54.0)
#define w3 (1.0/216.0)

#define sqrt2 1.4142135623730951 
#define sqrt3 1.7320508075688772

const double psi_gr_wv[PSI_NGRAD] = {w0,
		w3, w2, w3, 
		w2, w1, w2, 
		w3, w2, w3, 
		w2, w1, w2, 
		w1,     w1, 
		w2, w1, w2, 
		w3, w2, w3, 
		w2, w1, w2, 
		w3, w2, w3};     

const double psi_gr_rnorm[PSI_NGRAD] = {0.0,
		1.0/sqrt3, 1.0/sqrt2, 1.0/sqrt3, 
		1.0/sqrt2, 1.0, 1.0/sqrt2, 
		1.0/sqrt3, 1.0/sqrt2, 1.0/sqrt3, 
		1.0/sqrt2, 1.0, 1.0/sqrt2, 
		1.0,                  1.0, 
		1.0/sqrt2, 1.0, 1.0/sqrt2, 
		1.0/sqrt3, 1.0/sqrt2, 1.0/sqrt3, 
		1.0/sqrt2, 1.0, 1.0/sqrt2, 
		1.0/sqrt3, 1.0/sqrt2, 1.0/sqrt3};   

const double psi_gr_rcs2 = 3.0;
#endif

/*****************************************************************************
 *
 *  psi_electric_field
 *
 *  Return the electric field associated with the current potential.
 *
 *  The gradient of the potential is differenced as
 *      E_x = - (1/2) [ psi(i+1,j,k) - psi(i-1,j,k ]
 *  etc
 *
 *****************************************************************************/

int psi_electric_field(psi_t * psi, int index, double e[3]) {

  int xs, ys, zs;

  assert(psi);

  coords_strides(&xs, &ys, &zs);
#ifndef OLD_SHIT
  assert(0);
  int nsites = coords_nsites();
  e[X] = -0.5*(psi->psi[addr_rank0(nsites, index + xs)] - psi->psi[addr_rank0(nsites, index - xs)]);
  e[Y] = -0.5*(psi->psi[addr_rank0(nsites, index + ys)] - psi->psi[addr_rank0(nsites, index - ys)]);
  e[Z] = -0.5*(psi->psi[addr_rank0(nsites, index + zs)] - psi->psi[addr_rank0(nsites, index - zs)]);
#else
  e[X] = -0.5*(psi->psi[index + xs] - psi->psi[index - xs]);
  e[Y] = -0.5*(psi->psi[index + ys] - psi->psi[index - ys]);
  e[Z] = -0.5*(psi->psi[index + zs] - psi->psi[index - zs]);
#endif
  return 0;
}

/*****************************************************************************
 *
 *  psi_electric_field_d3qx
 *
 *  Return the electric field associated with the current potential.
 *
 *  The gradient of the potential is differenced with wider stencil 
 *  that includes nearest and next-to-nearest neighbours.
 *
 *****************************************************************************/

int psi_electric_field_d3qx(psi_t * psi, int index, double e[3]) {

  int p;
  int nsites;
  int coords[3], coords_nbr[3], index_nbr;
  double aux;

  assert(psi);

  nsites = coords_nsites();
  coords_index_to_ijk(index, coords);

  e[X] = 0;  
  e[Y] = 0;
  e[Z] = 0;

  for (p = 1; p < PSI_NGRAD; p++) {

    coords_nbr[X] = coords[X] + psi_gr_cv[p][X];
    coords_nbr[Y] = coords[Y] + psi_gr_cv[p][Y];
    coords_nbr[Z] = coords[Z] + psi_gr_cv[p][Z];

    index_nbr = coords_index(coords_nbr[X], coords_nbr[Y], coords_nbr[Z]);
#ifndef OLD_SHIT
    aux = psi_gr_wv[p]* psi_gr_rcs2 * psi->psi[addr_rank0(nsites, index_nbr)];
#else
    aux = psi_gr_wv[p]* psi_gr_rcs2 * psi->psi[index_nbr];
#endif
    e[X] -= aux * psi_gr_cv[p][X]; 
    e[Y] -= aux * psi_gr_cv[p][Y];  
    e[Z] -= aux * psi_gr_cv[p][Z];  

  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_grad_rho
 *
 *  Returns the gradient of the charge densities of each species.
 *
 *  The gradient of the potential is differenced as
 *    grad_rho_x = (1/2) [ psi->rho(i+1,j,k) - psi->rho(i-1,j,k)]
 *  etc.
 *
 *  If we are next to a boundary site we use one-sided derivatives.
 *
 *  Boundary to the left:
 *    grad_rho_x = -3/2*psi->rho(i,j,k)+2*psi->rho(i+1,k,j)-1/2*psi->rho(i+2,k,j)
 *  etc.
 *  Boundary to the right:
 *    grad_rho_x = 3/2*psi->rho(i,j,k)-2*psi->rho(i-1,k,j)+1/2*psi->rho(i-2,k,j)
 *  etc.
 *
 *****************************************************************************/

int psi_grad_rho(psi_t * psi,  map_t * map, int index, int n, double grad_rho[3]) {

  int nsites;
  int xs, ys, zs;
  int index0, index1;
  int status0, status1;

  assert(psi);
  assert(n < psi->nk);
  assert(grad_rho);

  nsites = coords_nsites();
  coords_strides(&xs, &ys, &zs);

  grad_rho[X] = 0.0;
  grad_rho[Y] = 0.0;
  grad_rho[Z] = 0.0;

  /* x-direction */
  index0 = index - xs;
  map_status(map, index0, &status0);
  index1 = index + xs;
  map_status(map, index1, &status1);
#ifndef OLD_SHIT
  assert(0);
  if (status0 == MAP_FLUID && status1 == MAP_FLUID){
    grad_rho[X] = 0.5*(psi->rho[addr_rank1(nsites, psi->nk, (index + xs), n)]
		     - psi->rho[addr_rank1(nsites, psi->nk, (index - xs), n)]);
  } 
  if (status0 != MAP_FLUID && status1 == MAP_FLUID){
    grad_rho[X] = - 1.5*psi->rho[addr_rank1(nsites, psi->nk, index, n)] 
      + 2.0*psi->rho[addr_rank1(nsites, psi->nk, (index + xs), n)]
      - 0.5*psi->rho[addr_rank1(nsites, psi->nk, (index + 2*xs), n)]; 
  }
  if (status0 == MAP_FLUID && status1 != MAP_FLUID){
    grad_rho[X] = + 1.5*psi->rho[addr_rank1(nsites, psi->nk, index, n)]
      - 2.0*psi->rho[addr_rank1(nsites, psi->nk, (index - xs), n)]
      + 0.5*psi->rho[addr_rank1(nsites, psi->nk, (index - 2*xs), n)];
  }
#else
  if (status0 == MAP_FLUID && status1 == MAP_FLUID){
    grad_rho[X] = 0.5*(psi->rho[psi->nk*(index + xs) + n] - 
		  psi->rho[psi->nk*(index - xs) + n]);
  } 
  if (status0 != MAP_FLUID && status1 == MAP_FLUID){
    grad_rho[X] = - 1.5*psi->rho[psi->nk*index + n] 
		  + 2.0*psi->rho[psi->nk*(index + xs) + n]
		  - 0.5*psi->rho[psi->nk*(index + 2*xs) + n]; 
  }
  if (status0 == MAP_FLUID && status1 != MAP_FLUID){
    grad_rho[X] = + 1.5*psi->rho[psi->nk*index + n]
                  - 2.0*psi->rho[psi->nk*(index - xs) + n]
                  + 0.5*psi->rho[psi->nk*(index - 2*xs) + n];
  }
#endif
  /* y-direction */
  index0 = index - ys;
  map_status(map, index0, &status0);
  index1 = index + ys;
  map_status(map, index1, &status1);
#ifndef OLD_SHIT
  if (status0 == MAP_FLUID && status1 == MAP_FLUID) {
    grad_rho[Y]
      = 0.5*(+ psi->rho[addr_rank1(nsites, psi->nk, (index+ys), n)]
	     - psi->rho[addr_rank1(nsites, psi->nk, (index-ys), n)]);
  }
  if (status0 != MAP_FLUID && status1 == MAP_FLUID) {
    grad_rho[Y]
      = - 1.5*psi->rho[addr_rank1(nsites, psi->nk, index, n)] 
      + 2.0*psi->rho[addr_rank1(nsites, psi->nk, (index + ys), n)]
      - 0.5*psi->rho[addr_rank1(nsites, psi->nk, (index + 2*ys), n)]; 
  }
  if (status0 == MAP_FLUID && status1 != MAP_FLUID) {
    grad_rho[Y] = + 1.5*psi->rho[addr_rank1(nsites, psi->nk, index, n)]
      - 2.0*psi->rho[addr_rank1(nsites, psi->nk, (index - ys), n)]
      + 0.5*psi->rho[addr_rank1(nsites, psi->nk, (index - 2*ys), n)];
  }
#else
  if (status0 == MAP_FLUID && status1 == MAP_FLUID){
    grad_rho[Y] = 0.5*(psi->rho[psi->nk*(index+ys) + n] - 
		  psi->rho[psi->nk*(index-ys) + n]);
  }
  if (status0 != MAP_FLUID && status1 == MAP_FLUID){
    grad_rho[Y] = - 1.5*psi->rho[psi->nk*index + n] 
		  + 2.0*psi->rho[psi->nk*(index + ys) + n]
		  - 0.5*psi->rho[psi->nk*(index + 2*ys) + n]; 
  }
  if (status0 == MAP_FLUID && status1 != MAP_FLUID){
    grad_rho[Y] = + 1.5*psi->rho[psi->nk*index + n]
                  - 2.0*psi->rho[psi->nk*(index - ys) + n]
                  + 0.5*psi->rho[psi->nk*(index - 2*ys) + n];
  }
#endif
  /* z-direction */
  index0 = index - zs;
  map_status(map, index0, &status0);
  index1 = index + zs;
  map_status(map, index1, &status1);
#ifndef OLD_SHIT
  if (status0 == MAP_FLUID && status1 == MAP_FLUID) {
    grad_rho[Z] = 0.5*(psi->rho[addr_rank1(nsites, psi->nk, (index+zs), n)] - 
		       psi->rho[addr_rank1(nsites, psi->nk, (index-zs), n)]);
  }
  if (status0 != MAP_FLUID && status1 == MAP_FLUID) {
    grad_rho[Z] = - 1.5*psi->rho[addr_rank1(nsites, psi->nk, index, n)] 
      + 2.0*psi->rho[addr_rank1(nsites, psi->nk, (index + zs), n)]
      - 0.5*psi->rho[addr_rank1(nsites, psi->nk, (index + 2*zs), n)]; 
  }
  if (status0 == MAP_FLUID && status1 != MAP_FLUID) {
    grad_rho[Z] = + 1.5*psi->rho[addr_rank1(nsites, psi->nk, index, n)]
      - 2.0*psi->rho[addr_rank1(nsites, psi->nk, (index - zs), n)]
      + 0.5*psi->rho[addr_rank1(nsites, psi->nk, (index - 2*zs), n)];
  }
#else
  if (status0 == MAP_FLUID && status1 == MAP_FLUID){
    grad_rho[Z] = 0.5*(psi->rho[psi->nk*(index+zs) + n] - 
		  psi->rho[psi->nk*(index-zs) + n]);
  }
  if (status0 != MAP_FLUID && status1 == MAP_FLUID){
    grad_rho[Z] = - 1.5*psi->rho[psi->nk*index + n] 
		  + 2.0*psi->rho[psi->nk*(index + zs) + n]
		  - 0.5*psi->rho[psi->nk*(index + 2*zs) + n]; 
  }
  if (status0 == MAP_FLUID && status1 != MAP_FLUID){
    grad_rho[Z] = + 1.5*psi->rho[psi->nk*index + n]
                  - 2.0*psi->rho[psi->nk*(index - zs) + n]
                  + 0.5*psi->rho[psi->nk*(index - 2*zs) + n];
  }
#endif
  return 0;
}


/*****************************************************************************
 *
 *  psi_grad_rho_d3qx
 *
 *  Returns the gradient of the charge densities of each species.
 *
 *  The gradient of the potential is differenced as
 *    grad_rho_x = (1/2) [ psi->rho(i+1,j,k) - psi->rho(i-1,j,k)]
 *  etc.
 *
 *  If we are next to a boundary site we use one-sided derivatives based on 
 *  Lagrange interpolating polynomials.
 *
 *  Boundary to the left:
 *    grad_rho_x = -3/2*psi->rho(i,j,k)+2*psi->rho(i+1,k,j)-1/2*psi->rho(i+2,k,j)
 *  etc.
 *  Boundary to the right:
 *    grad_rho_x = 3/2*psi->rho(i,j,k)-2*psi->rho(i-1,k,j)+1/2*psi->rho(i-2,k,j)
 *  etc.
 *
 *****************************************************************************/

int psi_grad_rho_d3qx(psi_t * psi,  map_t * map, int index, int n, double grad_rho[3]) {

  int p;
  int nsites;
  int coords[3], coords1[3], coords2[3]; 
  int index1, index2;
  int status, status1, status2;
  double aux;

  assert(psi);
  assert(n < psi->nk);
  assert(grad_rho);
  assert(0);

  nsites = coords_nsites();
  coords_index_to_ijk(index, coords);
  map_status(map, index, &status);

  grad_rho[X] = 0;  
  grad_rho[Y] = 0;
  grad_rho[Z] = 0;

  for (p = 1; p < PSI_NGRAD; p++) {

    coords1[X] = coords[X] + psi_gr_cv[p][X];
    coords1[Y] = coords[Y] + psi_gr_cv[p][Y];
    coords1[Z] = coords[Z] + psi_gr_cv[p][Z];

    index1 = coords_index(coords1[X], coords1[Y], coords1[Z]);
    map_status(map, index1, &status1);

    if(status == MAP_FLUID && status1 == MAP_FLUID) { 

      aux = psi_gr_wv[p]* psi_gr_rcs2 * psi->rho[psi->nk*index1 + n];

      grad_rho[X] += aux * psi_gr_cv[p][X]; 
      grad_rho[Y] += aux * psi_gr_cv[p][Y];  
      grad_rho[Z] += aux * psi_gr_cv[p][Z];  

    }

    else {

      coords1[X] = coords[X] - psi_gr_cv[p][X];
      coords1[Y] = coords[Y] - psi_gr_cv[p][Y];
      coords1[Z] = coords[Z] - psi_gr_cv[p][Z];
      index1 = coords_index(coords1[X], coords1[Y], coords1[Z]);
      map_status(map, index1, &status1);

      coords2[X] = coords[X] - 2*psi_gr_cv[p][X];
      coords2[Y] = coords[Y] - 2*psi_gr_cv[p][Y];
      coords2[Z] = coords[Z] - 2*psi_gr_cv[p][Z];
      index2 = coords_index(coords2[X], coords2[Y], coords2[Z]);
      map_status(map, index2, &status2);
	
      if(status == MAP_FLUID && status1 == MAP_FLUID && status2 == MAP_FLUID) {

        /* Subtract the above 'fluid' half of the incomplete two-point formula. */
        /* Note: subtracting means adding here because of inverse lattice vectors. */
#ifndef OLD_SHIT
	assert(0);
	aux = psi_gr_wv[p]* psi_gr_rcs2 * psi->rho[addr_rank1(nsites, psi->nk, index1, n)];
#else
	aux = psi_gr_wv[p]* psi_gr_rcs2 * psi->rho[psi->nk*index1 + n];
#endif
	grad_rho[X] += aux * psi_gr_cv[p][X]; 
	grad_rho[Y] += aux * psi_gr_cv[p][Y];  
	grad_rho[Z] += aux * psi_gr_cv[p][Z];  
#ifndef OLD_SHIT
        /* Use one-sided derivative instead */
	grad_rho[X] += psi_gr_wv[p] * psi_gr_rcs2 * 
	  (3.0*psi->rho[addr_rank1(nsites, psi->nk, index, n)] 
	   - 4.0*psi->rho[addr_rank1(nsites, psi->nk, index1, n)] 
	   + 1.0*psi->rho[addr_rank1(nsites, psi->nk, index2, n)]) 
	  * psi_gr_rnorm[p]* psi_gr_cv[p][X];

	grad_rho[Y] += psi_gr_wv[p] * psi_gr_rcs2 * 
	  (3.0*psi->rho[addr_rank1(nsites, psi->nk, index, n)] 
	   - 4.0*psi->rho[addr_rank1(nsites, psi->nk, index1, n)] 
	   + 1.0*psi->rho[addr_rank1(nsites, psi->nk, index2, n)]) 
	  * psi_gr_rnorm[p] * psi_gr_cv[p][Y];

	grad_rho[Z] += psi_gr_wv[p] * psi_gr_rcs2 * 
	  (3.0*psi->rho[addr_rank1(nsites, psi->nk, index, n)] 
	   - 4.0*psi->rho[addr_rank1(nsites, psi->nk, index1, n)] 
	   + 1.0*psi->rho[addr_rank1(nsites, psi->nk, index2, n)]) 
	  * psi_gr_rnorm[p] * psi_gr_cv[p][Z];
#else
        /* Use one-sided derivative instead */
	grad_rho[X] += psi_gr_wv[p] * psi_gr_rcs2 * 
			(3.0*psi->rho[psi->nk*index  + n] 
		       - 4.0*psi->rho[psi->nk*index1 + n] 
                       + 1.0*psi->rho[psi->nk*index2 + n]) 
		       * psi_gr_rnorm[p]* psi_gr_cv[p][X];

	grad_rho[Y] += psi_gr_wv[p] * psi_gr_rcs2 * 
			(3.0*psi->rho[psi->nk*index  + n] 
		       - 4.0*psi->rho[psi->nk*index1 + n] 
	               + 1.0*psi->rho[psi->nk*index2 + n]) 
		       * psi_gr_rnorm[p] * psi_gr_cv[p][Y];

	grad_rho[Z] += psi_gr_wv[p] * psi_gr_rcs2 * 
			(3.0*psi->rho[psi->nk*index  + n] 
                       - 4.0*psi->rho[psi->nk*index1 + n] 
	               + 1.0*psi->rho[psi->nk*index2 + n]) 
		       * psi_gr_rnorm[p] * psi_gr_cv[p][Z];
#endif
      }

    }

  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_grad_eps_d3qx
 *
 *****************************************************************************/

int psi_grad_eps_d3qx(f_vare_t fepsilon, int index, double grad_eps[3]) {

  int p;
  int coords[3], coords1[3]; 
  int index1;
  double aux, eps1;

  assert(fepsilon);
  assert(grad_eps);
#ifndef OLD_SHIT
  assert(0);
#endif
  coords_index_to_ijk(index, coords);

  grad_eps[X] = 0;  
  grad_eps[Y] = 0;
  grad_eps[Z] = 0;

  for (p = 1; p < PSI_NGRAD; p++) {

    coords1[X] = coords[X] + psi_gr_cv[p][X];
    coords1[Y] = coords[Y] + psi_gr_cv[p][Y];
    coords1[Z] = coords[Z] + psi_gr_cv[p][Z];

    index1 = coords_index(coords1[X], coords1[Y], coords1[Z]);
    fepsilon(index1, &eps1);

    aux = psi_gr_wv[p]* psi_gr_rcs2 * eps1;

    grad_eps[X] += aux * psi_gr_cv[p][X]; 
    grad_eps[Y] += aux * psi_gr_cv[p][Y];  
    grad_eps[Z] += aux * psi_gr_cv[p][Z];  

  }

  return 0;
}

