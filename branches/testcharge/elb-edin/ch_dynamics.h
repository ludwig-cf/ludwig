#ifndef _CH_DYNAMICS_H
#define _CH_DYNAMICS_H
#include "lb_types.h"


#define max(i,j) ((i) >= (j) ? (i) : (j))
void advect_ch( int xmax, int ymax, int zmax, double **conc_plus, double **conc_minus, 
  int ** inside, double **rho, double **jx, double **jy, double **jz );
int diffuse_euler( double ** c_plus, double ** c_minus, double ** phi_sor, int ** inside );
int diffuse_euler2( double ** c_plus, double ** c_minus, double ** phi_sor, int ** inside, real *** solute_force );
int diffuse_midpoint( double ** c_plus, double ** c_minus, double ** phi_sor, int ** inside );
void calc_fluxes( double ** phi, double ** c_plus, double ** c_minus, double ** flux_site_plus, double ** flux_site_minus, int ** inside, real scale );
void calc_fluxes2( double ** phi, double ** c_plus, double ** c_minus, double ** flux_site_plus, double ** flux_site_minus, int ** inside, real *** solute_force, real scale );
int move_charges( double ** c_plus, double ** c_minus, double ** flux_site_plus, double ** flux_site_minus, int ** inside );
void move_d_charges( double ** c_plus, double ** c_minus, double ** flux_site_plus, double ** flux_site_minus, int ** inside, int scale );
void sor(int lx, int ly, int lz, double **c_minus,
	 double **c_plus, double **phi);
void f_ext_colloid( object * obj, real **phi, real **c_plus, real **c_minus, int **inside, double ** phi_p_x, double ** phi_p_y, double ** phi_p_z, real *** solute_force );
void f_ext_colloid2( object * obj, real **phi, real **c_plus, real **c_minus, int **inside, double ** phi_p_x, double ** phi_p_y, double ** phi_p_z, real *** solute_force );
void calc_gradient( real **phi_sor, double ** phi_p_x, double ** phi_p_y, double ** phi_p_z );
void f_el_ext_colloids( object * obj, real **c_plus, real **c_minus, int **inside, double ** phi_p_x, double ** phi_p_y, double ** phi_p_z, real * tot_c_mom );
void f_el_ext_fluid( real **c_plus, real **c_minus, int **inside, double ** phi_p_x, double ** phi_p_y, double ** phi_p_z, real * tot_f_mom, real *** solute_force );
void inj_el_momentum( int **inside, real * tot_c_mom, real * tot_f_mom, object * obj, real *** solute_force );
void reshape3D( real ** l_old, real ** l_new );
void sor2( int lx, int ly, int lz, double **c_minus, double **c_plus, double **phi );
void sorPBC( real ** phi );
void calc_current( object * obj, real *** velcs_ptr, real **c_plus, real **c_minus, real * J );
void scale_ch_solid( int ** inside, real **c_plus, real **c_minus, real factor );

#endif /* CH_DYNAMICS */
