#ifndef _LATTICE_LB_H
#define _LATTICE_LB_H
#include "lb_types.h"
#include "simpar.h"

/* choosing c_s^2=\frac{1}{2}        */
                
#define a_01  1.0 / 12.0
#define a_02  1.0 / 24.0
#define a_11  1.0 / 6.0
#define a_12  1.0 / 12.0
#define a_21  1.0 / 4.0
#define a_22  1.0 / 8.0

#define NUMVEL 18
#define IN 1

/* SOLID - FLUID MODEL */
#define  SOLID 10
#define  FLUID 0
#define  WALL  2
#define  CORNER  3

/* Broken links model */
#define CONTINUOUS      0
#define BROKEN      (1<<l)
#define ALL_BROKEN   1022
#define FIXED       32768

/* N.B 1022 = 0000001111111110 */

#define x_num_bit 7
#define y_num_bit 7
#define z_num_bit 7

#define MAX_PTR 35000
#define MAX_OBJECTS 1000
#define MAX_BUFF 2000

void  lbe_initvel( real *** velcs_ptr, vector uu, int *site_type[]);
void  lbe_bconds( real *** velcs_ptr );
void  lbe_movexy( real *** velcs_ptr );
void  wallz_plus_bc( real *** velcs_ptr, int x_pos );
void  wallz_minus_bc( real *** velcs_ptr, int x_pos );
void  bnodes_f( real *** velcs_ptr, object * objects, object * objects_ptr, b_node * b_nodes_ptr, int n_obj );
real soft_sphere_force( real h );
real  do_z_column( real ** velcz_ptr, real nu, int x, int y, 
       vector * ptot, vector  *str, real   *solute_force[3], real *jx, 
       real   *jy, real   *jz, real   *rho, int * site_type );

void  init_delta( double * delta );
/*
void  lbe_initvel( real *velcs_ptr[][18], vector uu, int *site_type[]);
void  lbe_bconds( real *velcs_ptr[][18] );
void  lbe_movexy( real * velcs_ptr[][18] );
real  do_z_column( real *velcz_ptr[18], real nu, int x, int y, 
       vector * ptot, vector  *str, real   *solute_force[3], real *jx, 
       real   *jy, real   *jz, real   *rho, int * site_type );
void  wallz_plus_bc( real * velcs_ptr[][18], int x_pos );
void  wallz_minus_bc( real * velcs_ptr[][18], int x_pos );
void  bnodes_f( real *velcs_ptr[][18], object * objects_ptr, b_node * b_nodes_ptr, int n_obj );
*/
int update( object * objects_ptr, b_node * bnodes_ptr, int ** site_type, real ** c_plus, real ** c_minus, real *** velcs_ptr );
inline real pbc_dist( int i, int j, int k, int a, int b, int c );
void find_twin( int i, int j, int k, object * objects_ptr, vector r_jump, int * twin_i, int * twin_j, int * twin_k );
void pbc_pos( int * x, int * y, int * z );
inline void pbc_vec_dist( int i, int j, int k, int a, int b, int c, real * dist_x, real * dist_y, real * dist_z );
inline real rpbc_vec_dist( real i, real j, real k, real a, real b, real c, vector * v_dist );
void move_colloid( object * objects_ptr, b_node * bnodes_ptr, int ** site_type, real ** c_plus, real ** c_minus, real *** velcs_ptr, vector * r_jump );

#endif /* LATTICE_LB */
