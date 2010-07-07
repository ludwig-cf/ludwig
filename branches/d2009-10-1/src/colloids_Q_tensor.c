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
#include "io_harness.h"
#include "ran.h"
#include "lattice.h"
#include "util.h"
#include "colloids_Q_tensor.h"

struct io_info_t * io_info_scalar_q_;

#define PLANAR_ANCHORING 1

static int scalar_q_write(FILE * fp, const int i, const int j, const int k);
static int scalar_q_write_ascii(FILE *, const int, const int, const int);
static double scalar_order_parameter(double q[3][3]);

void COLL_set_Q(){

  int ia;
  int ic,jc,kc;
  
  Colloid * p_colloid;

  double r0[3];
  double rsite0[3];
  double normal[3];
  double dir[3];
  double dir_prev[3];

  Colloid * colloid_at_site_index(int);

  int nlocal[3],offset[3];
  int index;

  double q[3][3];
  double d[3],v[3][3];
  double director[3];
  double len_normal;
  double amplitude;
  int emax,enxt,nrots;
  double rdotd;
  double dir_len;
  amplitude = 0.33333333;

  coords_nlocal(nlocal);
  coords_nlocal_offset(offset);
  
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	
	index = coords_index(ic, jc, kc);

	p_colloid = colloid_at_site_index(index);
	
	/* check if this node is inside a colloid */
	if(p_colloid != NULL){
	  

	  /* Need to translate the colloid position to "local"
	   * coordinates, so that the correct range of lattice
	   * nodes is found */

	  r0[X] = p_colloid->s.r[X] - 1.0*offset[X];
	  r0[Y] = p_colloid->s.r[Y] - 1.0*offset[Y];
	  r0[Z] = p_colloid->s.r[Z] - 1.0*offset[Z];

	  rsite0[X] = 1.0*ic;
	  rsite0[Y] = 1.0*jc;
	  rsite0[Z] = 1.0*kc;

	  /* calculate the vector between the centre of mass of the
	   * colloid and node i, j, k 
	   * so need to calculate rsite0 - r0 */

	  coords_minimum_distance(r0, rsite0, normal);

	  /* now for homeotropic anchoring only thing needed is to
	   * normalise the the surface normal vector  */

	  len_normal = modulus(normal);
	  assert(len_normal <= p_colloid->s.ah);

	  if (len_normal < 10e-8) {
	    /* we are very close to the centre of the colloid.
	     * set the q tensor to zero */
	    q[X][X] = 0.0;
	    q[X][Y] = 0.0;
	    q[X][Z] = 0.0;
	    q[Y][Y] = 0.0;
	    q[Y][Z] = 0.0;
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
	  
	  jacobi(q,d,v,&nrots);
	  
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

	    dir[X] = v[X][emax];
	    dir[Y] = v[Y][emax];
	    dir[Z] = v[Z][emax];

	    /* calculate the projection of the director along the surface
	     * normal and remove that from the director to make the
	     * director perpendicular to the surface */

	    rdotd = dot_product(normal, dir)/(len_normal*len_normal);
	    
	    dir[X] = dir[X] - rdotd*normal[X];
	    dir[Y] = dir[Y] - rdotd*normal[Y];
	    dir[Z] = dir[Z] - rdotd*normal[Z];
	    
	    dir_len = modulus(dir);

	    if (dir_len < 10e-8) {
	      /* the vectors were [almost] parallel.
	       * now we use the direction of the previous node i,j,k
	       * this fails if we are in the first node, so not great fix...
	       */

	      cross_product(dir_prev, normal, dir);
	      dir_len = modulus(dir);
	      info("\n i,j,k, %d %d %d\n", ic,jc,kc);
	      assert(dir_len>10e-8);
	    }
	    assert(dir_len > 10e-8);

	    for (ia = 0; ia < 3; ia++) {
	      director[ia] = dir[ia] / dir_len;
	      dir_prev[ia] = dir[ia] / dir_len;
	    }

#else
	    /* Homeotropic anchoring */

	    director[X] = normal[X]/len_normal;
	    director[Y] = normal[Y]/len_normal;
	    director[Z] = normal[Z]/len_normal;
#endif
	    q[X][X] = 3.0/2.0*amplitude*(director[X]*director[X] - 1.0/3.0);
	    q[X][Y] = 3.0/2.0*amplitude*(director[X]*director[Y]);
	    q[X][Z] = 3.0/2.0*amplitude*(director[X]*director[Z]);
	    q[Y][Y] = 3.0/2.0*amplitude*(director[Y]*director[Y] - 1.0/3.0);
	    q[Y][Z] = 3.0/2.0*amplitude*(director[Y]*director[Z]);
	    
	    phi_set_q_tensor(index, q);

	}
	
      }
    }
  }
  return;
}

void COLL_randomize_Q(double delta_r){
  
  int ic,jc,kc;
  
  Colloid * p_colloid;
  Colloid * colloid_at_site_index(int);

  int nlocal[3],offset[3];
  int index;
  
  double Pi;
  double q[3][3];
  double amplitude,phase1,phase2;
  
  /* set amplitude to something small */
  amplitude = 0.0000001;
  
  coords_nlocal(nlocal);
  coords_nlocal_offset(offset);
  
  Pi = 4.0*atan(1.0);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	
	index = coords_index(ic, jc, kc);

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

#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);\
	a[k][l]=h+s*(g-h*tau);
#define n 3
void jacobi(double (*a)[n], double d[], double (*v)[n], int *nrot)
{
  int j,iq,ip,i;
  double tresh,theta,tau,t,sm,s,h,g,c;
  double b[n],z[n];

  for (ip=0;ip<n;ip++) {
    for (iq=0;iq<n;iq++) v[ip][iq]=0.0;
    v[ip][ip]=1.0;
  }
  for (ip=0;ip<n;ip++) {
    b[ip]=d[ip]=a[ip][ip];
    z[ip]=0.0;
  }
  *nrot=0;
  for (i=1;i<=50;i++) {
    sm=0.0;
    for (ip=0;ip< n-1;ip++) {
      for (iq=ip+1;iq<n;iq++)
	sm += fabs(a[ip][iq]);
    }
    if (sm == 0.0) {
      return;
    }
    if (i < 4)
      tresh=0.2*sm/(n*n);
    else
      tresh=0.0;
    for (ip=0;ip<n-1;ip++) {
      for (iq=ip+1;iq<n;iq++) {
	g=100.0*fabs(a[ip][iq]);
	if (i > 4 && (fabs(d[ip])+g) == fabs(d[ip])
	    && (fabs(d[iq])+g) == fabs(d[iq]))
	  a[ip][iq]=0.0;
	else if (fabs(a[ip][iq]) > tresh) {
	  h=d[iq]-d[ip];
	  if ((fabs(h)+g) == fabs(h))
	    t=(a[ip][iq])/h;
	  else {
	    theta=0.5*h/(a[ip][iq]);
	    t=1.0/(fabs(theta)+sqrt(1.0+theta*theta));
	    if (theta < 0.0) t = -t;
	  }
	  c=1.0/sqrt(1+t*t);
	  s=t*c;
	  tau=s/(1.0+c);
	  h=t*a[ip][iq];
	  z[ip] -= h;
	  z[iq] += h;
	  d[ip] -= h;
	  d[iq] += h;
	  a[ip][iq]=0.0;
	  for (j=0;j<=ip-1;j++) {
	    ROTATE(a,j,ip,j,iq)
	      }
	  for (j=ip+1;j<=iq-1;j++) {
	    ROTATE(a,ip,j,j,iq)
	      }
	  for (j=iq+1;j<n;j++) {
	    ROTATE(a,ip,j,iq,j)
	      }
	  for (j=0;j<n;j++) {
	    ROTATE(v,j,ip,j,iq)
	      }
	  ++(*nrot);
	}
      }
    }
    for (ip=0;ip<n;ip++) {
      b[ip] += z[ip];
      d[ip]=b[ip];
      z[ip]=0.0;
    }
  }
  printf("Too many iterations in routine jacobi");
  exit(0);
}
#undef n
#undef ROTATE

/*****************************************************************************
 *
 *  scalar_q_io_init
 *
 *  Initialise the I/O information for the scalar order parameter
 *  output related to blue phase tensor order parameter.
 *
 *  This stuff lives here until a better home is found.
 *
 *****************************************************************************/

void scalar_q_io_init(void) {

  /* Use a default I/O struct */
  io_info_scalar_q_ = io_info_create();

  io_info_set_name(io_info_scalar_q_, "Scalar order parameter");
  io_info_set_write_binary(io_info_scalar_q_, scalar_q_write);
  io_info_set_write_ascii(io_info_scalar_q_, scalar_q_write_ascii);
  io_info_set_bytesize(io_info_scalar_q_, 1*sizeof(double));

  io_info_set_format_ascii(io_info_scalar_q_);
  // io_info_set_format_binary(io_info_scalar_q_);
  io_write_metadata("scalar_q", io_info_scalar_q_);

  return;
}

/*****************************************************************************
 *
 *  scalar_q_write_ascii
 *
 *  Write the value of the scalar order parameter at (ic, jc, kc).
 *
 *****************************************************************************/

static int scalar_q_write_ascii(FILE * fp, const int ic, const int jc,
				const int kc) {
  int index, n;
  double q[3][3];
  double qs;
  double d[3],v[3][3];
  int nrots,emax,enxt;
  int i;
  Colloid * colloid_at_site_index(int);
  Colloid * p_colloid;

  index = coords_index(ic, jc, kc);
  phi_get_q_tensor(index, q);

  /* JUHO: YOU NEED CODE TO WORK OUT VALUE REQUIRED HERE */
  jacobi(q,d,v,&nrots);
  
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
  
  qs = d[emax]; 

  //assert(qs<= 1.0/3.0);
  // if(qs >= 1.0/3.0){
  // info("\n qs = %lf, i, j, k, %d, %d, %d \n", qs, ic, jc, kc);
  //}

  p_colloid = colloid_at_site_index(index);
  
  if(p_colloid != NULL){
    /* if inside colloid print perfect order */
    qs=0.33333333333;
  }
  n = fprintf(fp, "%4d %4d %4d %22.15e ", ic,jc,kc,qs);
  if (n < 37) fatal("fprintf(qs) failed at index %d\n", index);
  
  if(p_colloid != NULL){
    /* this column is printed 0.0 if inside colloid otherwise the same as previous */
    qs=0.0;
      }
  n = fprintf(fp, "%22.15e ", qs);
  if (n < 23) fatal("fprintf(qs) failed at index %d\n", index);
  
  for (i=0;i<3;i++){
    qs=v[i][emax];
    n = fprintf(fp, "%22.15e ", qs);
    if (n < 23) fatal("fprintf(qs) failed at index %d\n", index);
  }
  n = fprintf(fp, "\n");
  if (n != 1) fatal("fprintf(qs) failed at index %d\n", index);

  return n;
}

/*****************************************************************************
 *
 *  colloids_fix_swd
 *
 *  The velocity gradient tensor used in the Beris-Edwards equations
 *  requires some approximation to the velocity at lattice sites
 *  inside particles. Here we set the lattice velocity based on
 *  the solid body rotation u = v + Omega x rb
 *
 *****************************************************************************/

void colloids_fix_swd(void) {

  int ic, jc, kc, index;
  int nlocal[3];
  int noffset[3];
  const int nextra = 1;

  double u[3];
  double rb[3];
  double x, y, z;

  Colloid * p_c;
  Colloid * colloid_at_site_index(int);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    x = noffset[X] + ic;
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      y = noffset[Y] + jc;
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {
	z = noffset[Z] + kc;

	index = coords_index(ic, jc, kc);

	p_c = colloid_at_site_index(index);

	if (p_c) {
	  /* Set the lattice velocity here to the solid body
	   * rotational velocity */

	  rb[X] = p_c->s.r[X] - x;
	  rb[Y] = p_c->s.r[Y] - y;
	  rb[Z] = p_c->s.r[Z] - z;

	  cross_product(p_c->s.w, rb, u);

	  u[X] += p_c->s.v[X];
	  u[Y] += p_c->s.v[Y];
	  u[Z] += p_c->s.v[Z];

	  hydrodynamics_set_velocity(index, u);

	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  scalar_q_write
 *
 *  Write scalar order parameter in binary.
 *
 *****************************************************************************/

static int scalar_q_write(FILE * fp, const int ic, const int jc,
			  const int kc) {
  int index, n;
  double q[3][3];
  double qs;

  index = coords_index(ic, jc, kc);
  phi_get_q_tensor(index, q);
  qs = scalar_order_parameter(q);

  n = fwrite(&qs, sizeof(double), 1, fp);
  if (n != 1) fatal("fwrite(qs) failed at index %d\n", index);

  return n;
}

/*****************************************************************************
 *
 *  scalar_order_parameter
 *
 *  Return the value of the scalar order parameter for given Q tensor.
 *
 *****************************************************************************/

double scalar_order_parameter(double q[3][3]) {

  int nrots, emax;
  double d[3], v[3][3];

  jacobi(q, d, v, &nrots);
  
  /* Find the largest eigenvalue */

  if (d[0] > d[1]) {
    emax=0;
  }
  else {
    emax=1;
  }
  if (d[2] > d[emax]) {
    emax=2;
  }

  return d[emax];
}
