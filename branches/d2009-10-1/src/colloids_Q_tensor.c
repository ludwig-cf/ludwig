/*
 * colloids_Q_tensor.c
 * routine to set the Q tensor inside a colloid to correspond
 * to homeotropic or planar anchoring at the surface
 * 11/11/09
 * -Juho
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "pe.h"
#include "build.h"
#include "coords.h"
#include "colloids.h"
#include "phi.h"
#include "colloids_Q_tensor.h"

#define PLANAR_ANCHORING 1

//extern FVector COLL_fcoords_from_ijk(int, int, int);
//extern FVector   COLL_fvector_separation(FVector, FVector);

void COLL_set_Q(){
  
  int i,j,k;
  int ic,jc,kc;
  
  Colloid * p_colloid;

  FVector r0;
  FVector rsite0;
  FVector normal;
  FVector dir;
  FVector COLL_fvector_separation(FVector, FVector);
  FVector COLL_fcoords_from_ijk(int, int, int);
  Colloid * colloid_at_site_index(int);

  int nlocal[3],offset[3];
  int index;

  double q[3][3];
  double m[3][3],d[3],v[3][3];
  double director[3];
  double len_normal,len_dir;
  double amplitude;
  int emax,enxt;
  double rdotd;
  double dir_len;
  amplitude = 0.33333333;

  get_N_local(nlocal);
  get_N_offset(offset);
  
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc < nlocal[Z]; kc++) {
	
	index = get_site_index(ic, jc, kc);

	p_colloid = colloid_at_site_index(index);
	
	/* check if this node is inside a colloid */
	if(p_colloid != NULL){
	  
	  /* get the colloid position 
	   */
	  
	  r0 = p_colloid->r;
	  
	  /* Need to translate the colloid position to "local"
	   * coordinates, so that the correct range of lattice
	   * nodes is found */
	  r0.x -= (double) offset[X];
	  r0.y -= (double) offset[Y];
	  r0.z -= (double) offset[Z];
	  
	  /* rsite0 is the coordinate position of the site */
	  rsite0 = COLL_fcoords_from_ijk(ic, jc, kc);
	  
	  /* calculate the vector between the centre of mass of the colloid and node i, j, k 
	   * so need to calculate rsite0 - r0
	   */
	  normal = COLL_fvector_separation(r0, rsite0);
	  /* now for homeotropic anchoring only thing needed is to normalise the
	     the surface normal vector
	  */
	  len_normal = sqrt(UTIL_dot_product(normal, normal));

	  if(len_normal < 10e-8){
	    /* we are very close to the centre of the colloid.
	     * set the q tensor to zero
	     */
	    q[0][0]=q[0][1]=q[0][2]=q[1][1]=q[1][2]=0.0;
	    phi_set_q_tensor(index,q);
	    continue;
	      }
	  
#if PLANAR_ANCHORING
	  /* now we need set the director inside the colloid
	     perpendicular to the vector normal of of the surface [i.e. it is
	     confined in a plane] i.e.  
	     perpendicular to the vector r between the centre of the colloid
	     corresponding node i,j,k
	     -Juho
	  */
	  
	  phi_get_q_tensor(index, q);
	  
	  //jacobi(q,d,v,&nrots);
	  
	  /* find the largest eigen value and corresponding eigen vector */
	  if (d[0] > d[1]) {
	      emax=0;
	      enxt=1;
	    }
	    else {
	      emax=1;
	      enxt=0;
	    }
	    if (d[2] > d[emax]) {
	      emax=2;
	    }
	    else if (d[2] > d[enxt]) {
	      enxt=2;
	    }
	    dir.x = v[0][emax];
	    dir.y = v[1][emax];
	    dir.z = v[2][emax];

	    /* calculate the projection of the director along the surface normal 
	     * and remove that from the director to make the director perpendicular to 
	     * the surface
	     */
	    rdotd = UTIL_dot_product(normal,dir)/(len_normal*len_normal);
	    
	    dir.x = dir.x - rdotd*normal.x;
	    dir.y = dir.y - rdotd*normal.y;
	    dir.z = dir.z - rdotd*normal.z;
	    
	    dir_len = sqrt(UTIL_dot_product(dir,dir));
	    assert(dir_len<10e-8);
	    
	    director[0] = dir.x/dir_len;
	    director[1] = dir.y/dir_len;
	    director[2] = dir.z/dir_len;
#else
	    /* Homeotropic anchoring */
	    director[0] = normal.x/len_normal;
	    director[1] = normal.y/len_normal;
	    director[2] = normal.z/len_normal;

#endif
	    q[0][0] = amplitude*(director[0]*director[0] - 1.0/3.0);
	    q[0][1] = amplitude*(director[0]*director[1]);
	    q[0][2] = amplitude*(director[0]*director[2]);
	    q[1][1] = amplitude*(director[1]*director[1] - 1.0/3.0);
	    q[1][2] = amplitude*(director[1]*director[2]);
	    
	    phi_set_q_tensor(index, q);
	}
	
      }
    }
  }
  return;
}

void COLL_randomize_Q(double delta_r){
  
  int i,j,k;
  int ic,jc,kc;
  
  Colloid * p_colloid;

  FVector r0;
  FVector rsite0;

  FVector COLL_fvector_separation(FVector, FVector);
  FVector COLL_fcoords_from_ijk(int, int, int);
  Colloid * colloid_at_site_index(int);

  int nlocal[3],offset[3];
  int index;
  
  double Pi;
  double q[3][3];
  double amplitude,phase1,phase2;
  
  /* set amplitude to something small */
  amplitude = 0.0000001;
  
  get_N_local(nlocal);
  get_N_offset(offset);
  
  Pi = 4.0*atan(1.0);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc < nlocal[Z]; kc++) {
	
	index = get_site_index(ic, jc, kc);

	p_colloid = colloid_at_site_index(index);
	
	/* check if this node is inside a colloid */
	if(p_colloid != NULL){
	  phase1= 2.0/5.0*Pi*(0.5-ran_parallel_uniform());
	  phase2= Pi/2.0+Pi/5.0*(0.5-ran_parallel_uniform());

	  q[X][X] = amplitude* (3.0/2.0*sin(phase2)*sin(phase2)*cos(phase1)*cos(phase1)-1.0/2.0);
	  q[X][Y] = 3.0*amplitude/2.0*(sin(phase2)*sin(phase2)*cos(phase1)*sin(phase1));
	  q[X][Z] = 3.0*amplitude/2.0*(sin(phase2)*cos(phase2)*cos(phase1));
	  q[Y][X] = q[X][Y];
	  q[Y][Y] = amplitude*(3.0/2.0*sin(phase2)*sin(phase2)*sin(phase1)*sin(phase1)-1.0/2.0);
	  q[Y][Z] = 3.0*amplitude/2.0*(sin(phase2)*cos(phase2)*sin(phase1));
	  q[Z][X] = q[X][Z];
	  q[Z][Y] = q[Y][Z];
	  q[Z][Z] = - q[X][X] - q[Y][Y];

	  phi_set_q_tensor(index, q);
	}
      }
    }
  }
}
