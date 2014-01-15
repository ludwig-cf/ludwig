/* lb code for a simple ideal fluid in*/
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "lb.h"
#include "io.h"
#include "init.h"
#include "simpar.h"
#include "ch_dynamics.h"
#include "mc_equilibration.h"

extern int x_dir[18], y_dir[18], z_dir[18];
extern int weighting[18];

sim_params_t params;

int main(int argc, char ** argv)
{
  int     n_step, n_obj;
  int     x, y, i, index, max_index;
  clock_t c0, c1; 
  c0 = clock();
  
  b_node * b_nodes = calloc( MAX_OBJECTS, sizeof(b_node) );
  object * objects = calloc( MAX_OBJECTS, sizeof(object) );

  vector ptot, str;

  extern int fluid_nodes;

  if( argc != 2 ){ printf("Usage: %s inputfile\n", argv[0]); exit(0); } 
  FILE * cfg_file;
  cfg_file = fopen(argv[1], "r"); 
  init_param( cfg_file, &params);
  fclose(cfg_file);

  extern double delta[18];
  init_delta( delta );
  
  /*parameter nu in collision that ensures that the previously prescribed kinematic viscosity is recovered */
  params.nu       = ( 6.0 * params.nu - 1.0 ) / ( 6.0 * params.nu + 1.0 );
  int siz_x    = params.Nx + 2; 
  int siz_y = params.Ny + 2;

  max_index = siz_x * siz_y;

  real ** phi_sor     = mem_r_lattice2( max_index, params.Nz + 2 ); 
  real ** c_plus      = mem_r_lattice2( max_index, params.Nz ); 
  real ** c_minus     = mem_r_lattice2( max_index, params.Nz ); 
  real ** rho         = mem_r_lattice2( max_index, params.Nz ); 
  real ** jx          = mem_r_lattice2( max_index, params.Nz ); 
  real ** jy          = mem_r_lattice2( max_index, params.Nz ); 
  real ** jz          = mem_r_lattice2( max_index, params.Nz ); 
  real ** phi_prime_x         = mem_r_lattice2( max_index, params.Nz + 2 ); 
  real ** phi_prime_y         = mem_r_lattice2( max_index, params.Nz + 2 ); 
  real ** phi_prime_z         = mem_r_lattice2( max_index, params.Nz + 2 ); 

  int ** site_type       = mem_i_lattice2( max_index, params.Nz ); 

  real *** velcs_ptr = alloc_3mtx( 18 );
  real *** solute_force = alloc_3mtx( 3 );

  init_wall( site_type );

  /* places the colloids in the system */
  int mixture = 0;
  if ( params.num_obj == 1 ){
    /* conf_gen puts particles at random: check they don't overlap with the walls */
    conf_gen( objects, &params.num_obj, params.r_eff );
  }else if( params.num_obj == 2 ){
    conf_gen2( objects, params.r_eff, params.r_pair );
  }else{
    mixture = conf_fromfile( objects, params.r_eff );
  }
  printf("Conf gen\n");

  for( n_obj = 0; n_obj < params.num_obj; n_obj++ )
    objects_init( &objects[n_obj], n_obj ); 
	/* initialize the node list of the colloid+additional required information */
  for( n_obj = 0; n_obj < params.num_obj; n_obj++ )
    bnodes_init( &objects[n_obj], &b_nodes[n_obj], objects[n_obj].self, site_type );

  /* Init charge distribution */
  if( !mixture ){ 
    init_ch_dist( params.Nx, params.Ny, params.Nz, site_type, params.Nx * params.Ny * params.Nz, c_minus, c_plus, (real **) phi_sor, objects);
  }else{
    init_ch_mixture( site_type, c_minus, c_plus , objects );
  }
  //init_ch_dist_NoCorners( site_type, c_minus, c_plus, objects);
  double tot_charge[2];
  check_charge_fluid( c_plus, c_minus, site_type, tot_charge );
  printf(" ONLY FLUID TOTAL CHARGE(+|-) INIT IS %e %e\n", tot_charge[0], tot_charge[1] );


  real res[10];
  check_conservations( velcs_ptr, site_type, c_plus, c_minus, res, objects );
  PRint_INIT;

  /* variable used in initialization to give the fluid a uniform velocity NOT WORKING NOW AS u_ampl hardcoded to 0.0 in lbe_initvel */
  vector u_ampl;
  u_ampl.x = params.u_x; u_ampl.y = params.u_y; u_ampl.z = params.u_z;
  /* initialization of velocity profile */
  lbe_initvel( velcs_ptr, u_ampl, site_type );
  /* uniform external field applied to the liquid */
  extern vector ff_ext;
  ff_ext.x = 1.0 * 6 * PIG * params.r_eff * u_ampl.x / (real) fluid_nodes;
  ff_ext.y = 1.0 * 6 * PIG * params.r_eff * u_ampl.y / (real) fluid_nodes;
  ff_ext.z = 1.0 * 6 * PIG * params.r_eff * u_ampl.z / (real) fluid_nodes;

  PRint_INIT2;
#ifdef NUM
  printf("In this elb version external force on colloid are added at numerator of formula (14) JCP Lowe Frenkel Masters\n");
#endif
  printf("FORCING ELECTRIC FIELD SLOPE dE_x %e\n\n", params.e_slope_x );


  /* GG INIT ENDED. NOW EQUILIBRATION */
  int in_multisteps = params.multisteps;
  real in_D_plus    = params.D_plus;
  real in_D_minus   = params.D_minus;
  real slope_x, slope_y, slope_z;
  int ch_equilibration, eqloops;
  char bindump[128] = "Eq_Ch_nofield_";
  /* Back up value of field as given in input file */
  slope_x = params.e_slope_x;
  slope_y = params.e_slope_y;
  slope_z = params.e_slope_z;

  if( strcmp(params.restart, "no") ){
    load_bin_lattice( params.restart, velcs_ptr, site_type, c_plus, c_minus, phi_sor, phi_prime_x, phi_prime_y, phi_prime_z );
    printf( "\n\tRESTARTING ---> Loading %s file\n\n", params.restart );
    params.multisteps = 1;
    params.D_plus  = params.D_therm;
    params.D_minus = params.D_therm;
    printf( "\n\tRestarting and setting D_\\pm thermal to %e and multistep to 1\n\n", params.D_therm );
  }else{
    /* FIRST EQUILIBRATION OF CHARGES. NO FIELD NO FLOW */
    /* Relax charges to equilibrium profiles (No Field). Fluid not used here */
    /* During charges equilibration we use multistep = 1 and D_\pm = D_therm */
    params.multisteps = 1;
    printf( "\n\tSetting D_\\pm to %e and multistep to 1\n\n", params.D_therm );

    /* Remove ext elect field for first part of eq */
    if ( (params.e_slope_x + params.e_slope_y + params.e_slope_z) != 0.0 ){
      params.e_slope_x = params.e_slope_y = params.e_slope_z = 0.0;
    }
    printf(" *****   Removing external field to %f %f %f: will begin with 1st long SOR  *****\n\n", 
        params.e_slope_x, params.e_slope_y, params.e_slope_z );
    if( !params.GreenKubo ) write_lattice( velcs_ptr, site_type, c_plus, c_minus, phi_sor, phi_prime_x, phi_prime_y, phi_prime_z, -3, params.output );
    sor2( params.Nx,params.Ny,params.Nz, c_minus, c_plus, phi_sor );
    fflush( NULL );

    ch_equilibration = 0;
    eqloops = 0;
    /* At the very very beginning charges move big time as they are faaar from equilibrium EDL */
    real in_diffusion_acc = params.diffusion_acc;
    params.D_plus = params.D_minus = params.D_therm;
    printf( "\n\tSetting starting thermal diffusion of D*\\pm diffusion to input value %e %e and diffusion accuracy %e\n", params.D_plus, params.D_minus, params.diffusion_acc );
    int sig = 0;
    while( !ch_equilibration ){
      printf( "\nEntering charge equilibration loop number %d with zero elec field\n", eqloops );
      eqloops++;
      ch_equilibration = diffuse_euler( c_plus, c_minus, phi_sor, site_type );
      if( !(eqloops % 50) ){
        //write_lattice( velcs_ptr, site_type, c_plus, c_minus, phi_sor, phi_prime_x, phi_prime_y, phi_prime_z, eqloops, params.output );
      }
      /* To exit from equilibration, not only you need convergence of charge diffusion to set input accuracy, but 
       * also that diffusion coefficients are greater than a threshold */
      if( ch_equilibration == 1 && params.D_therm <= 0.001 ){ 
        printf( "\n\tEquilibrium to %e accuracy of charges using diffusions (+,-) %e,%e reached!\n", params.diffusion_acc, params.D_plus, params.D_minus );
        params.D_therm *= 2.0;
        params.D_plus *= 2.0; params.D_minus *= 2.0;
        printf( "\tNow increasing thermal diffusion: \
        setting D*\\pm diffusion to %e %e\n", params.D_plus, params.D_minus );
        ch_equilibration = 0; 
      }else if( params.D_therm > 0.001 && params.D_therm <= 0.01 && sig == 0 ){
        printf( "\n\tEquilibrium to %e accuracy of charges using diffusions (+,-) %e,%e reached!\n", params.diffusion_acc, params.D_plus, params.D_minus );
        printf( "\tSetting ACCURACY to input value %e\n", in_diffusion_acc );
        printf( "\tSetting max_eq_fluxes to 10000 so that it is not unlikely to rescale diffusions\n" );
        params.diffusion_acc = in_diffusion_acc;
        params.max_eq_fluxes = 1000000.0;
        ch_equilibration = 0; 
        sig = 1;
      }
      fflush( NULL );
    }
    printf( "Charge diffusion WITHOUT electric field converged with accuracy %e at loop %d\n", params.diffusion_acc, eqloops );
    printf( "Dumping binfile\n" );
    strcat( bindump, params.output );
    dump_bin_lattice( velcs_ptr, site_type, c_plus, c_minus, phi_sor, phi_prime_x, phi_prime_y, phi_prime_z, 0, bindump );
  } // End if restart
  if( !params.GreenKubo ) write_lattice( velcs_ptr, site_type, c_plus, c_minus, phi_sor, phi_prime_x, phi_prime_y, phi_prime_z, -2, params.output );

  if( !params.GreenKubo ){
    /* SECOND EQUILIBRATION OF CHARGES. NO FLOW */
    /* Relax charges to equilibrium re-adding external field. Fluid not used here */
    params.e_slope_x = slope_x;
    params.e_slope_y = slope_y;
    params.e_slope_z = slope_z;
    printf("\n *****   External field switched on: %f %f %f  *****\n\n", params.e_slope_x, params.e_slope_y, params.e_slope_z );
    ch_equilibration = 0;
    eqloops = 0;
    while( !ch_equilibration ){
      printf( "\nEntering charge equilibration loop number %d with Electric Field\n", eqloops );
      eqloops++;
      ch_equilibration = diffuse_euler( c_plus, c_minus, phi_sor, site_type );
      fflush( NULL );
    }
    printf( "Charge equilibration WITH electric field converged with accuracy %e at loop %d\n", params.diffusion_acc, eqloops );
    printf( "Dumping binfile\n" );
    strcpy( bindump, "Eq_Ch_field_" );
    strcat( bindump, params.output );
    dump_bin_lattice( velcs_ptr, site_type, c_plus, c_minus, phi_sor, phi_prime_x, phi_prime_y, phi_prime_z, 0, bindump );

    check_conservations( velcs_ptr, site_type, c_plus, c_minus, res, objects );
    printf( "\nAFTER CHARGE EQUILIBRATION: Tot fluid mass %f Tot charge %e (%e/%e) Fluid (also in colloid) momentum %e %e %e \
      and Colloid momentum %e %e %e \nDELTA MOM = %e %e %e\n", res[0], res[1], res[8], res[9], res[2], res[3], res[4], 
      res[5], res[6], res[7], res[2] + res[5], res[3] + res[6], res[4] + res[7] );
    printf("\n");
    if( !params.GreenKubo ) write_lattice( velcs_ptr, site_type, c_plus, c_minus, phi_sor, phi_prime_x, phi_prime_y, phi_prime_z, -1, params.output );
  }else{ // GREEN KUBO SETUP
    /* Adding perturbation (see equations) */
    write_lattice( velcs_ptr, site_type, c_plus, c_minus, phi_sor, phi_prime_x, phi_prime_y, phi_prime_z, 888, params.output );
    init_GK( velcs_ptr, &objects[0], c_plus, c_minus, site_type, rho, jx, jy, jz );
    /* Make sure there is no ext elect field */
    params.e_slope_x = params.e_slope_y = params.e_slope_z = 0.0;
    /* Make sure frequency dumping of colloid velocity is 1 */
    params.outfreq = 1;
    check_conservations( velcs_ptr, site_type, c_plus, c_minus, res, objects );
    printf( "\nGREENKUBO of Step %d: Tot fluid mass %f Tot charge %e (%e/%e) Fluid (also in colloid) momentum %e %e %e \
        and Colloid momentum %e %e %e \nDELTA MOM = %e %e %e\n", n_step, res[0], res[1], res[8], res[9], res[2], 
        res[3], res[4], res[5], res[6], res[7], res[2] + res[5], res[3] + res[6], res[4] + res[7] );
    /* Output stuff */
    if( !strcmp(params.restart, "no") ){ printf("Cannot run a GreenKubo setup without restarting"); exit( 0 ); }
    write_lattice( velcs_ptr, site_type, c_plus, c_minus, phi_sor, phi_prime_x, phi_prime_y, phi_prime_z, 777, params.output );
  } // end if restart or 1st nofield equilibration

  printf("\n\n\t\t********************************\n");
  printf("\t\t**   FLUID-CHARGES DYNAMICS   **\n");
  printf("\t\t********************************\n");
  printf("\n");
  fflush( NULL );

  /* GG EQUIL ENDED: NOW DYNAMICS */
  /* Resetting multistepping and diffusions as in input file */
  params.multisteps = in_multisteps;
  params.D_plus     = in_D_plus;
  params.D_minus    = in_D_minus;
  printf( "\n\tSetting back D_\\pm to input values %e %e and multistep to %d\n\n", params.D_plus, params.D_minus, params.multisteps );

  slope_x = slope_y = slope_z = 0.0;
  real vel_coll_old = 0.0;
  int flag = 1;
  for( n_step = 0; n_step <= params.numsteps; n_step++ ){
    if( flag == 0 ) printf( "t = %d\n", n_step );
    if ( n_step == 0 && (params.e_slope_x + params.e_slope_y + params.e_slope_z) != 0.0 ){
      printf("\n\n *****  FIRST STAGE OF DYNAMICS SETTING TO ZERO EXTERNAL ELECTRIC FIELD  *****\n");
      printf(" *****  This will last %d timesteps  *****\n\n",  params.numsteps / 20 );
      slope_x = params.e_slope_x;
      slope_y = params.e_slope_y;
      slope_z = params.e_slope_z;
      params.e_slope_x = params.e_slope_y = params.e_slope_z = 0.0;
    }
    if ( n_step == (params.numsteps / 20) && (slope_x + slope_y + slope_z) != 0.0 ){
      printf("\n\n *****  SECOND STAGE OF DYNAMICS SWITCHING ON EXTERNAL ELECTRIC FIELD  *****\n");
      if( flag ){ 
        n_step = 0;
        printf("Reset n_step = 0\n");
        flag = 0;
      }
      params.e_slope_x = slope_x;
      params.e_slope_y = slope_y;
      params.e_slope_z = slope_z;
      printf(" *****   External field is %f %f %f  *****\n\n", params.e_slope_x, params.e_slope_y, params.e_slope_z );
    }

    /* Bounce back if there is a wall (assumed here in z-direction) */
    if( params.width > 0 ){
      wallz_plus_bc( velcs_ptr, params.width );	  
      wallz_minus_bc( velcs_ptr, params.Nz-params.width - 1 );
    }			  
    
    /* Charge Advection */
    advect_ch( params.Nx, params.Ny, params.Nz, c_plus, c_minus, site_type, rho, jx, jy, jz );
    sor2( params.Nx, params.Ny, params.Nz, c_minus, c_plus, phi_sor );
    printf( "\nStarted elb loop @@@ time %d: Advected charges\n", n_step );

    /* Charge Diffusion (multi-timestepping) */
    for( i = 1; i <= params.multisteps; i++ ){
      if( params.multisteps > 1 ) printf("t = %d in loop %d multistep\n", n_step, i);
      /* Note that ch_equilibration that will be checked in while will be the one given
       * by the last diffusion run in multitimestep. It makes sense, as should not converge
       * in the middle of timestep but not at the end, i.e. last diffusion_* run */
      //ch_equilibration = diffuse_midpoint( c_plus, c_minus, phi_sor, site_type );
      ch_equilibration = diffuse_euler( c_plus, c_minus, phi_sor, site_type );
      //ch_equilibration = diffuse_euler2( c_plus, c_minus, phi_sor, site_type, solute_force );
    }
    fflush( NULL );

    /* Calc electrostatic forces on colloid and solute using gradient */
    calc_gradient( phi_sor, phi_prime_x, phi_prime_y, phi_prime_z );
    real tot_c_mom[3], tot_f_mom[3];
    f_el_ext_colloids( objects, c_plus, c_minus, site_type, phi_prime_x, phi_prime_y, phi_prime_z, tot_c_mom );
    /* GG COMMENT if using LinkFlux diffuse_euler2. f_el_ext_fluid as now solute force is calculated with LF in diffuse_euler when calculating fluxes */
    f_el_ext_fluid( c_plus, c_minus, site_type, phi_prime_x, phi_prime_y, phi_prime_z, tot_f_mom, solute_force );
    /* It does not matter if fluid is pushed externally or not */
    /* GG COMMENT injection removed now in order not to add compensating momentum when using euler2 */
    inj_el_momentum( site_type, tot_c_mom, tot_f_mom, objects, solute_force );

    /* Apply periodic boundary conditions */
    lbe_bconds( velcs_ptr ); 

    /* Colloid forces: bnodes_f has to go before LB part as new coll velocity has to be known in the algorithm to correctly calculate bounce back */
    if( params.num_obj > 0 ){
      if( mod( n_step, 2 ) == 0 ){ 
	      for( n_obj = 0; n_obj < params.num_obj; n_obj++ ){
	        bnodes_f( velcs_ptr, objects, &objects[n_obj], &b_nodes[n_obj], n_obj );
	      }
      }else{
        for( n_obj = params.num_obj - 1; n_obj >= 0; n_obj-- ){
	        bnodes_f( velcs_ptr, objects, &objects[n_obj], &b_nodes[n_obj], n_obj );
	      }
      }
    }
    /* LB Advection step: you displace the pointer of the vector in the x-y plane */   
    lbe_movexy( velcs_ptr );
    
    for( x = 1; x <= params.Nx; x++){
      for( y = 1; y <= params.Ny; y++ ){
       	index = x * siz_y + y;
	      /* Collision step: here one advects in the z direction the pointer strutcture 
         * and performs the relaxation part of LB as well */
	      /* the variable str is not currently used. ptot is not very relevant either */
	      do_z_column( velcs_ptr[index], params.nu, x, y, &ptot, &str, solute_force[index], 
        jx[index], jy[index], jz[index],rho[index], site_type[index] );
      }
    }

    /* Update colloids: update also redoes list in case it moved and we chose to move */ 
    int moved = 0;
    for( n_obj = 0; n_obj < params.num_obj; n_obj++ ){
      moved = update( &objects[n_obj], &b_nodes[n_obj], site_type, c_plus, c_minus, velcs_ptr );
      if( moved ){
        moved = 0;
        write_lattice( velcs_ptr, site_type, c_plus, c_minus, phi_sor, phi_prime_x, phi_prime_y, phi_prime_z, n_step, params.output );
        check_electroneutrality( c_plus, c_minus, site_type );
        /* Recalculate potential to sync it */
        sor2( params.Nx, params.Ny, params.Nz, c_minus, c_plus, phi_sor );
      }
    }

    if( (n_step % params.outfreq) == 0 || moved ){
      double res[10];
      real acceleration = ( objects[0].u.x - vel_coll_old ) / objects[0].u.x;
      printf("\nColloid 0 @ end of t= %d in %f %f %f, centered in %d %d %d, velocity %e (norm accel=%e%%) %e %e\n",  
          n_step, objects[0].r.x, objects[0].r.y, objects[0].r.z, objects[0].i.x, objects[0].i.y, objects[0].i.z, 
          objects[0].u.x, acceleration, objects[0].u.y, objects[0].u.z);
      vel_coll_old = objects[0].u.x;
      if( !params.GreenKubo ){
        write_lattice( velcs_ptr, site_type, c_plus, c_minus, phi_sor, phi_prime_x, phi_prime_y, phi_prime_z, n_step, params.output );
        //sprintf( bindump, "RUN_%s", params.output  );
        //dump_bin_lattice( velcs_ptr, site_type, c_plus, c_minus, phi_sor, phi_prime_x, phi_prime_y, phi_prime_z, n_step, bindump );
      }
      real J[3];
      if( params.GreenKubo ){
        calc_current( objects, velcs_ptr, c_plus, c_minus, J );
      }
      write_colloids( objects, J, n_step, params.output );
      check_conservations( velcs_ptr, site_type, c_plus, c_minus, res, objects );
      printf( "\nEND of Step %d: Tot fluid mass %f Tot charge %e (%e/%e) Fluid (also in colloid) momentum %e %e %e \
          and Colloid momentum %e %e %e \nDELTA MOM = %e %e %e\n", n_step, res[0], res[1], res[8], res[9], res[2], 
          res[3], res[4], res[5], res[6], res[7], res[2] + res[5], res[3] + res[6], res[4] + res[7] );
      printf("\n\n");
    }
    fflush( NULL );
  } /* End n_step loop */
  if( params.GreenKubo ){
    write_lattice( velcs_ptr, site_type, c_plus, c_minus, phi_sor, phi_prime_x, phi_prime_y, phi_prime_z, n_step, params.output );
  }

  struct tm *ptr;
  time_t tm;
  tm = time(NULL);
  ptr = localtime( &tm );
  printf("\n\t\tENDING elb rv %s SIMULATION @ %s\n", SVN_REV,  asctime(ptr));
  c1 = clock();
  printf( "Simulation took %.1f seconds\n", ((double)( c1 - c0 )) / CLOCKS_PER_SEC );
  return( 1 );
}
