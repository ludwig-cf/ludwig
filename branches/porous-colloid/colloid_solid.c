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

	  if(h<0.0)fatal("Stopping");
	  
	  for (ia = 0; ia < 3; ia++) {
	    rhat[ia] = r12[ia]/rsep;
	  }

	  fmod = soft_sphere_force(h);

	  for (ia = 0; ia < 3; ia++) {
	    pc->force[ia] += fmod*rhat[ia];
	  }
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

/******************************************************************************
 *
 *  cylinder_lubrication
 *
 * This gives the normal lubrication correction for colloid of hydrodynamic
 * radius ah at position near a cylinder wall (circular confinement in 2D)
 * Top and bottom are taken care of by wall lubrication.
 *
 * The correction is based on sphere-wall correction:
 * wall_lubrication (in wall.c)
 *
 *
 *****************************************************************************/
  
double cylinder_lubrication(const int dim, const double r[3], const double ah) {
  
  double force;
  double hlub;
  double h[3], hlength, gap;
  double centre[3];
  double eta;
  int i;
  double lubrication_rcnormal_ = 0.5;
  
  double cylinder_radius_ = 11.0;
  int long_axis_ = Z; /*X, Y or Z */

  force = 0.0;
  hlub = lubrication_rcnormal_;
  eta = get_eta_shear();
  
  if (dim != long_axis_ ) {
    hlength = 0.0;
    
    for (i = 0; i < 3; i++ ) {
      if (i == long_axis_ )continue;
      
      centre[i] = Lmin(i) + 0.5*L(i);
      h[i] = centre[i] - r[i];
      hlength += h[i] * h[i];
      
      /* we want the the positive component 
       * ie. opposite to the velocity direction
       */
      h[i] = fabs(h[i]);
    }
    
    hlength = sqrt(hlength);
    gap = cylinder_radius_ - (hlength + ah);
    
    assert(gap > 0.0);
    
    if (gap < hlub) {
      force = -6.0*pi_*eta*ah*ah*(1.0/gap - 1.0/hlub)*h[dim]/hlength;
    }
  }

  return force;
}
    
    
    
