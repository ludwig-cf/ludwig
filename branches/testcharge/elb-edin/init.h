#ifndef _INIT_H
#define _INIT_H
#include "lb_types.h"

real *** alloc_3mtx( int n );
real ** mem_r_lattice2( int max_index, int max_z );
int ** mem_i_lattice2( int max_index, int max_z );
real ** mem_r_lattice( int max_index, int max_z );
int   ** mem_i_lattice( int max_index, int max_z );
void free_r_lattice( real ** ptr, int num_ptrs );
void free_2D( void ** ptr );
void free_3D( void *** ptr );
//void init_ch( charge_type * ch );
void init_GK( real *** velcs_ptr, object * obj, real ** c_plus, real ** c_minus, int ** site_type, real ** rho, real ** jx, real ** jy, real ** jz );
void init_wall( int ** site_type );

#define min(i,j) ((i) <= (j) ? (i) : (j))
#define nint(i)  ((i) >=  0.0  ? (int)((i)+(0.5)) : (int)((i)-(0.5)))
#define mod(i,j) (i) - (i)/(j)*(j)

#define PRint_INIT   printf( "\nEND INIT: Tot fluid mass %f Tot charge %e (%e/%e) Fluid (also in colloid) momentum \
      %e %e %e and Colloid momentum %e %e %e \nDELTA MOM = %e %e %e\n\n", res[0], res[1], res[8], \
      res[9], res[2], res[3], res[4], res[5], res[6], res[7], res[2] + res[5], res[3] + res[6], \
      res[4] + res[7] );

#define PRint_INIT2    printf( "\nMass of each colloid %e and total CN %d\n", objects[0].mass, params.Nx * params.Ny * params.Nz -  fluid_nodes ); \
printf("EXTERNAL MOMENTUM M_x M_y M_z(i.e. increasing m[123] in collision) on each fluid node %e %e %e (total in box %e %e %e)\n", ff_ext.x, ff_ext.y, ff_ext.z, ff_ext.x * fluid_nodes, ff_ext.y * fluid_nodes, ff_ext.z * fluid_nodes);

void  conf_gen(  object *objects, int *n_obj, real rsq);
void  conf_gen2(  object *objects, real r_eff, real r_pair );
int   conf_fromfile(  object *objects, real r_eff );
void  objects_init( object  *objects_ptr, int n_obj);
void bnodes_init( object *objects_ptr, b_node *bnodes_ptr, int coll_label, int *site_type[] );
int disk( real x, real y, real z, object *objects_ptr );
int disk2d( real y, real z, object *objects_ptr );
void init_ch_dist(int lx,int ly, int lz, int ** site_type, int  totsit, double **c_minus, double **c_plus, real ** el_potential, object *objects);
void init_ch_mixture( int ** site_type, double ** c_minus, double ** c_plus, object * objects );
void init_ch_dist_NoCorners( int ** site_type, double **c_minus, double **c_plus, object *objects);
inline real phi_edl_sphere( int i, int j, int k, real charge_sph, object obj );

float gasdev(long *idum);
float ran1(long *idum); 
int   nint2( float x);

#endif /* INIT */
