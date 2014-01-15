#ifndef _LB_TYPES_H
#define _LB_TYPES_H

/************************/
/* SIMULATION PRECISION */
/************************/

typedef double real;

#define real double 

#define TRUE -1
#define FALSE 0

typedef struct{
  real  x;
  real  y;
  real  z;
} vector;
/*
typedef struct CHARGED_PAR
{
  real bjl, sigma, lambda_D, pi;
  real anormf0, tmpvar_phi;
  int totsit;
  int D_iter;
  real D_plus, D_minus, Dtot;
  real elec_slope_x,  elec_slope_y,  elec_slope_z, gap;
  int numit, Tprint_eq, Tprint_run;
} charge_type;
*/
typedef struct{
  int   *i_nodes;
  int   *int_nodes;
  real *r_x;
  real *r_y;
  real *r_z;
  int   num_bnode;
  int   num_int_nodes;
  int   vol;
  real *u_nodes; 
} b_node;

typedef struct{
    int     self;
    int     (*func_ptr)();
    real  mass;
    real  charge;
    real  inertia;
    real  mass_fac;
    int    mass_flag;
    int    max_bnode;
    struct size{
             int   x;
             int   y;
             int   z;
           } size_m, 
             size_p,
             *relat;
    int      n_points;  
    struct icoord{
             int    x;
             int    y;
             int    z;
           } i;
    struct vector{
             real  x;
             real  y;
             real  z;
            } r,
              u,
              w,
              f,
              t,
              f_ext,
              t_ext;         
   real    rad;
   real    r_sq;
   real    theta;
} object;

#endif /* LB_TYPES */
