/*****************************************************************************
 *
 *  colloid_solid_lubrication.c
 *
 *  Responsible for colloid-solid lubrication forces
 *  
 *
 *  $Id: colloid_solid_lubrication.c $
 *
 *  Juho
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "util.h"
#include "physics.h"
#include "colloid_solid.h"
#include "model.h"
#include "site_map.h"
#include "potential.h"

static int solid_lubrication_force(const double r[3], const double ah, double force[3]);
static void lubrication_sphere_sphere_wall(const double a1, const double a2,
				    const double h, const double hcut,
				    const double rhat[3], double f[3]);

static double wall_sphere_radius_ = 0.5;
static double wall_sphere_lubrication_cutoff_ = 0.25;

void solid_lubrication(colloid_t * pc, double force[3]) {
  
  int i, j, k;
  int i_min, i_max;
  int j_min, j_max;
  int k_min, k_max;
  int index, ia;
  double r0[3], rsite[3];
  double rsep, r12[3];
  int N[3];
  int offset[3];
  double a1, a2;
  double h, hcut;
  double rhat[3];
  double force12[3];
  char status;
  double fmod;
  
  coords_nlocal(N);
  coords_nlocal_offset(offset);
  
  force[0] = 0.0;
  force[1] = 0.0;
  force[2] = 0.0;

  /* constants */
  a1 = pc->s.ah;
  a2 = wall_sphere_radius_;
  hcut = wall_sphere_lubrication_cutoff_;

  /* centre of the colloid in local coordinates */
  r0[X] = pc->s.r[X] - 1.0*offset[X];
  r0[Y] = pc->s.r[Y] - 1.0*offset[Y];
  r0[Z] = pc->s.r[Z] - 1.0*offset[Z];

  /* cube aroung the colloid */
  i_min = imax(1,    (int) floor(r0[X] - a1 - a2));
  i_max = imin(N[X], (int) ceil (r0[X] + a1 + a2));
  j_min = imax(1,    (int) floor(r0[Y] - a1 - a2));
  j_max = imin(N[Y], (int) ceil (r0[Y] + a1 + a2));
  k_min = imax(1,    (int) floor(r0[Z] - a1 - a2));
  k_max = imin(N[Z], (int) ceil (r0[Z] + a1 + a2));

  for (i = i_min; i <= i_max; i++) {
    for (j = j_min; j <= j_max; j++) {
      for (k = k_min; k <= k_max; k++) {

	index = coords_index(i, j, k);
	status = site_map_get_status_index(index);
	
	if (status != BOUNDARY) continue;
	
	rsite[X] = 1.0*i;
	rsite[Y] = 1.0*j;
	rsite[Z] = 1.0*k;
	coords_minimum_distance(rsite, r0, r12);
	
	rsep = modulus(r12);
	h = rsep - a1 - a2;

	/*sanity check */
	if (h < hcut ){
	  /*
	  printf("mur mur: (dit ours finlandaise)\n");
	  printf("horror error h < 0.0\n");
	  printf("colloid id: %d\n",pc->s.index);
	  printf("%lf %lf %lf %lf\n",h, rsep, a1, a2);
	  printf("solid: %d %d %d\n", i, j, k);
	  printf("colloid: %lf %lf %lf\n", r0[0], r0[1], r0[2]);
	  printf("colloid: %lf %lf %lf\n", pc->s.r[0], pc->s.r[1], pc->s.r[2]);
	  printf("r12: %lf %lf %lf\n", r12[0], r12[1], r12[2]);
	  printf("vc: %lf %lf %lf\n", pc->s.v[0],pc->s.v[1],pc->s.v[2]);
	  */
	  if(h<0.0)fatal("Stopping");
	  
	  //if (h < hcut ) {
	  //printf("hip\n");
	  for (ia = 0; ia < 3; ia++) {
	    rhat[ia] = r12[ia]/rsep;
	  }
	  fmod = 0.0;
	  fmod = soft_sphere_force(h);
	  /*printf("f: %lf %lf\n",fmod, h);*/
	  for (ia = 0; ia < 3; ia++) {
	    pc->force[ia] += fmod*rhat[ia];
	    force[ia] += fmod*rhat[ia];
	    /*printf("fc[%d]: %lf\n", ia, fmod*rhat[ia]);*/
	  }
	  
	  //lubrication_sphere_sphere_wall(a1, a2, h, hcut, rhat, force12);
	  //printf("fc: %lf %lf %lf\n", force12[0],force12[1],force12[2]);
	  //for (ia = 0; ia < 3; ia++ ) {
	  //force[ia] += force12[ia];
	  //}
	  
	  //}
	}
      }
    }
  }
  return;
}
	
void lubrication_sphere_sphere_wall(const double a1, const double a2,
				    const double h, const double hcut,
				    const double rhat[3], double f[3]) {

  int ia;
  double eta;
  double fmod;
  double hr, hcr;
  
  eta = get_eta_shear();
  hr = 1.0/h;
  hcr = 1.0/hcut;

  fmod = -6.0*pi_*eta*a1*a1*a2*a2*(hr - hcr)/((a1 + a2)*(a1 + a2));

  for (ia = 0; ia < 3; ia++ ) {
    f[ia] = fmod*rhat[ia];
  }
  
  return;
}
 
 

/*

  while (p_link) {
    //printf("link status %d\n",p_link->status);
    //printf("cv: %d %d %d\n",cv[p_link->p][X],cv[p_link->p][Y],cv[p_link->p][Z]);
    if (p_link->status == LINK_BOUNDARY) {
*/
      /* for (i = 0; i < 3; i++ ){
	force_link[i] = 0.0;
      }
      */

      //along_axis = 0;

      /* consider only links along coordinate axis */
      //if (cv[p_link->p][X] == 1 && cv[p_link->p][Y] == 0 && cv[p_link->p][Z] == 0 ) {
	/* along x */
	//along_axis = 1;
//}
//    else if (cv[p_link->p][X] == 0 && cv[p_link->p][Y] == 1 && cv[p_link->p][Z] == 0 ) {
	/*along y */
/*
	along_axis = 1;
      }
      else if (cv[p_link->p][X] == 0 && cv[p_link->p][Y] == 0 && cv[p_link->p][Z] == 1 ) {*/
	/* along z */
/*
	along_axis = 1;
      }
      
      test = 0;
      along_axis = 1;
      if (along_axis == 1 ) {
	
	coords_index_to_ijk(p_link->i, isite);
	for (ia = 0; ia < 3; ia++) {
	  rsite[ia] = 1.0*isite[ia]*cv[p_link->p][ia];
	  r0[ia] = (p_link->rb[ia] - 1.0*offset[ia])*cv[p_link->p][ia];
	  printf("r0[ia] cv[ia] %lf %d %d\n", r0[ia], cv[p_link->p][ia],ia);
	}
	
	coords_minimum_distance(r0, rsite, rsep);
	
	test = solid_lubrication_force(r0 ,pc->s.ah, force_link);
      
	//if (test == 1){
	  printf("error: r < ah\n");
	  printf("colloid: %d\n", pc->s.index);
	  printf("x,y,z: %lf %lf %lf\n",pc->s.r[0],pc->s.r[1],pc->s.r[2]);
	  printf("x,y,z: %lf %lf %lf\n",r0[0], r0[1], r0[2]);
	  printf("x,y,z: %lf %lf %lf\n",p_link->rb[0], p_link->rb[1], p_link->rb[2]);
	  printf("vx,vy,vz: %lf %lf %lf\n",pc->s.v[0],pc->s.v[1],pc->s.v[2]);
	  printf("cv: %d %d %d\n",cv[p_link->p][X],cv[p_link->p][Y],cv[p_link->p][Z]);
	  coords_index_to_ijk(p_link->i, isite);
	  printf("link: %d %d %d\n", isite[0], isite[1], isite[2]);
	  coords_index_to_ijk(p_link->j, isite);
	  printf("inside: %d %d %d\n", isite[0], isite[1], isite[2]);
	  if(test==1)exit(0);
	  //}
	for (ia = 0; ia < 3; ia++ ){
	  force[ia] += force_link[ia];
	}  
      }
      }*/
    /* Next link */
/*
    p_link = p_link->next;
  }
  return ;
}
*/

int solid_lubrication_force(const double r[3], const double ah, double force[3]) {
  
  double hlub;
  double h,rl;
  int i;
  
  /*hlub = lubrication_rcnormal_;*/
  hlub = 0.5;

  h = 0.0;
  rl = 0.0;
  for (i = 0; i < 3; i++) {
    force[i] = 0.0;
    rl += r[i]*r[i];
  }
  rl = sqrt(rl);
  if ( rl < ah){
    printf("HORROR: %lf, %lf\n", rl ,ah);
    return 1;
  }
  h = rl - ah;
  /*printf("info: h %lf %lf\n",h, rl);*/
  if (h < hlub) {
    for (i = 0; i < 3; i++) {
      force[i] = 6.0*pi_*get_eta_shear()*ah*ah*(1.0/h - 1.0/hlub)*r[i]/rl;
    }
    printf("f: %lf, %lf, %lf %lf\n",force[0],force[1],force[2],h);

  }
  return 0;
}


  
