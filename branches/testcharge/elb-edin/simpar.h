#ifndef _SYMPAR_H
#define _SYMPAR_H
#include <stdio.h>
#include "lb_types.h"

/*******************/
/* INFILE HANDLING */
/*******************/

#define ENDSTRING '\0'
#define LINESIZ   128
#define PARSIZ    64
#define TABSIZ    64
#define BLANK     "blank"

#define PIG 3.1415927

struct param {
  char name[PARSIZ];      
  char value[PARSIZ];      
};

typedef struct param param_t;

struct sim_params {
  int numsteps;
  int Nx;
  int Ny;
  int Nz;
  long seed;
  real wall_vel;
  real u_x;
  real u_y;
  real u_z;
  int width;
  real diffusion_acc;
  int num_obj;
  real r_eff;
  real rho0;
  real nu;
  real kbt;
  real ch_bjl;
  real ch_wall;
  real ch_lambda;
  real e_slope_x;
  real e_slope_y;
  real e_slope_z;
  real D_therm;
  real D_plus;
  real D_minus;
  real f_accuracy;
  real poisson_acc;
  real max_eq_fluxes;
  real r_pair;
  int GreenKubo;
  int multisteps;
  char restart[64];
  int outfreq;
  char output[64];
  real ch_objects;
  char ch_distype[1];
  char inf_colloids[64];
  int  move;
};

typedef struct sim_params sim_params_t;

int check_param( param_t param, char * listparams[] );
int set_params( sim_params_t * sim_params, param_t * table_params, char * name_params[] );
int init_param( FILE *cfg_file, sim_params_t * sim_params );
void lookup_param( char * p_name, param_t * table_params, char * p_value );

#endif /* SYMPAR */
