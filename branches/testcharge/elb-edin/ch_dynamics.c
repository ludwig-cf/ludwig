#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lb.h"
#include "init.h"
#include "simpar.h"
#include "ch_dynamics.h"

extern sim_params_t params;

extern int x_dir[18], y_dir[18], z_dir[18];
extern float a0[18];
extern float a1[18];
extern double delta[18]; 
extern int fluid_nodes;

FILE * file_pot;

void advect_ch( int xmax, int ymax, int zmax, double **conc_plus, double **conc_minus, 
  int ** inside, double **rho, double **jx, double **jy, double **jz )
{
    int i,j,k,index,indexp,index0,index1,index00,index10;     /* loop variables */
    int ip,jp,kp;    /* loop variables */
    double flux_link_minus, flux_link_plus;
    /* double delta=1.0; delta is the lattice spacing */
    double vx, vy, vz;
    double rho_here;
    double ax, ay, az;
    int ix,iy,iz;
    double conc_plus_old, conc_plus_new, conc_minus_old, conc_minus_new;

    int max_index = ( params.Nx + 2 ) * ( params.Ny + 2 );
    real ** flux_site_plus   = mem_r_lattice2( max_index, params.Nz );
    real ** flux_site_minus  = mem_r_lattice2( max_index, params.Nz );

    int siz_y = params.Ny + 2;
    // GG ugly workaround for now
    xmax = params.Nx;
    ymax = params.Ny;
    zmax = params.Nz;
    
    /* Compute the old concentrations and set the fluxes to zero */
    conc_plus_old = 0.0;
    conc_minus_old = 0.0;
    for(i=1;i<=xmax;i++){
      for(j=1;j<=ymax;j++){
	index=i*siz_y+j;
	for(k=0;k<zmax;k++){
	  conc_plus_old += conc_plus[index][k];
	  conc_minus_old += conc_minus[index][k];
	  flux_site_minus[index][k]=0.0;
	  flux_site_plus[index][k]=0.0;
	}
      }
    }
    /* THIS IS VERY CRIPTIC...*/
    /* What I am doing is what discussed with you.
     * Basically (when using this word it means that the explanation is a little bit mistic :-) )
    * you have to compute the portion of the volume which goes in one of the neighbouring 26 cells.
    * The part of the volume that goes in the next cell is ax*ay*az.
    * ax could have three values.
    * ax = (1 - vx)
    * ax = vx
    * ax = 0 (zero)
    * If I multiply by 0 I do not continue.
    * One complication of the algorithm is that if ix (the x coordinate of the next cell) is 0,
    * ax is 0 but I should transport on y and z direction.
    *
    * Forget this explanation. It works!!!!
    */
        
    for(i=1;i<=xmax;i++){
      for(j=1;j<=ymax;j++){
	index=i*siz_y+j;
	for(k=0;k<zmax;k++){
	  if(inside[index][k]==FLUID){
	    rho_here= rho[index][k] ;
	    vx = jx[index][k] / rho_here;
	    vy = jy[index][k] / rho_here;
	    vz = jz[index][k] / rho_here;

	    for(ix=-1;ix<=1;ix++){
	      for(iy=-1;iy<=1;iy++){
		for(iz=-1;iz<=1;iz++){
		  ip = i+ix;
		  if(ip>xmax)ip=1;
		  if(ip<1) ip = xmax;
		  jp = j+iy;
		  if(jp>ymax)jp=1;
		  if(jp<1) jp = ymax;
		  kp=k+iz;
		  if(kp>zmax-1) kp=0;
		  if(kp<0) kp=zmax-1;
		  indexp=ip*siz_y+jp;
		  if(inside[indexp][kp]==FLUID){
		    ax = vx*ix;
		    ay = vy*iy;
		    az = vz*iz;
		    if( ax >= 0.0 && ay >= 0.0 && az >=0.0 ){
		      if(ix==0) ax = 1.0 - fabs(vx);
		      if(iy==0) ay = 1.0 - fabs(vy);
		      if(iz==0) az = 1.0 - fabs(vz);
		      
		      flux_link_plus  = ax*ay*az*conc_plus[index][k];
		      flux_link_minus = ax*ay*az*conc_minus[index][k];
		      
		    }else{
		      flux_link_plus  = 0.0;
		      flux_link_minus = 0.0;
		    }
		    flux_site_plus[index] [k]   -= flux_link_plus;
		    flux_site_plus[indexp][kp]  += flux_link_plus;
		    flux_site_minus[index] [k]  -= flux_link_minus;
		    flux_site_minus[indexp][kp] += flux_link_minus;
		  } 
		}
	      }
	    }   
	  }   
	}
      }
    }
    /* Update the concentrations */
    for(i=1;i<=xmax;i++){
      for(j=1;j<=ymax;j++){
       	index=i*siz_y+j;
	for(k=0;k<zmax;k++){
	  if(inside[index][k]==FLUID){
	    conc_plus[index][k]  += flux_site_plus[index][k];
	    conc_minus[index][k] += flux_site_minus[index][k];
	  }
	}   
      }   
    }
    for(j=1;j<=ymax;j++){
      index0=0*siz_y+j;
      index1=xmax*siz_y+j;
      index00=(xmax+1)*siz_y+j;
      index10=1*siz_y+j;
      for(k=0;k<zmax;k++){
	conc_plus[index0][k]   = conc_plus[index1][k];
	conc_minus[index0][k]  = conc_minus[index1][k];
	conc_plus[index00][k]  = conc_plus[index10][k];
	conc_minus[index00][k] = conc_minus[index10][k];
      }
    }   

    for(i=1;i<=xmax;i++){
      index0=i*siz_y+0;
      index1=i*siz_y+ymax;
      index00=i*siz_y+(ymax+1);
      index10=i*siz_y+1;
      for(k=0;k<zmax;k++){
	conc_plus[index0][k]   = conc_plus[index1][k];
	conc_minus[index0][k]  = conc_minus[index1][k];
	conc_plus[index00][k]  = conc_plus[index10][k];
	conc_minus[index00][k] = conc_minus[index10][k];
      }
    } 
    conc_plus_new = conc_minus_new = 0.0;
    for(i=1;i<=xmax;i++){
      for(j=1;j<=ymax;j++){
	index=i*siz_y+j;
	for(k=0;k<zmax;k++){
	  conc_plus_new += conc_plus[index][k];
	  conc_minus_new += conc_minus[index][k];
	}
      }
    }
    if(fabs(conc_minus_new-conc_minus_old)>10e-8){
	printf("Total concentration has changed!\n");
	printf("conc_minus_old = %e conc_minus_new = %e \n", conc_minus_old, conc_minus_new);
    }
    if(fabs(conc_plus_new-conc_plus_old)>10e-6){
      printf("Total concentration has changed!\n");
      printf("conc_old = %e  %e conc_new = %e  %e \n", conc_minus_old, conc_plus_old, conc_minus_new, conc_plus_new);
    }
  free_2D( (void **) flux_site_plus );
  free_2D( (void **) flux_site_minus );
}

int diffuse_midpoint( double ** c_plus, double ** c_minus, double ** phi_sor, int ** inside )
{
  int max_index = ( params.Nx + 2 ) * ( params.Ny + 2 ); 
  real ** flux_site_plus   = mem_r_lattice2( max_index, params.Nz );
  real ** flux_site_minus  = mem_r_lattice2( max_index, params.Nz );
  real ** c_plus_try       = mem_r_lattice2( max_index, params.Nz );
  real ** c_minus_try      = mem_r_lattice2( max_index, params.Nz );

  // Assuming potential anch charges in sync 
  // Calculation of fluxes at t (divided by 2.0 as required by midpoint method)
  calc_fluxes( phi_sor, c_plus, c_minus, flux_site_plus, flux_site_minus, inside, 2.0 );
  // Move a copy of charges (to keep old ones) using half flux calc at time t
  memcpy( (void *) c_plus_try[0],  (void *) c_plus[0],  max_index * params.Nz * sizeof(real) );
  memcpy( (void *) c_minus_try[0], (void *) c_minus[0], max_index * params.Nz * sizeof(real) );
  // Calculate charges at midpoint t + dt / 2 (store in different place)
  move_charges( c_plus_try, c_minus_try, flux_site_plus, flux_site_minus, inside );
  // Calculate (fake) potential at midpoint. Here phi_try not needed as this potential likely is
  // more similar to the final one that has to be computed by sor when charges are updated
  sor2( params.Nx, params.Ny, params.Nz, c_minus, c_plus, phi_sor );
  // Calculate fluxes at midpoint (with charges and potential at midpoint)
  calc_fluxes( phi_sor, c_plus_try, c_minus_try, flux_site_plus, flux_site_minus, inside, 1.0 );
  // Update charges from t to t + dt O(h^3)
  int equilibrated = move_charges( c_plus, c_minus, flux_site_plus, flux_site_minus, inside );
  // Now run sor to keep in sync charges and potential at t + dt
  sor2( params.Nx, params.Ny, params.Nz, c_minus, c_plus, phi_sor );

  free_2D( (void **) c_plus_try );
  free_2D( (void **) c_minus_try );
  free_2D( (void **) flux_site_plus );
  free_2D( (void **) flux_site_minus );
  return equilibrated;
}

void calc_fluxes( double ** phi_sor, double ** c_plus, double ** c_minus, double ** flux_site_plus, double ** flux_site_minus, int ** inside, real scale )
{
  double D_plus, D_minus;
  D_plus  = params.D_plus  / params.multisteps;
  D_minus = params.D_minus / params.multisteps;
  
  // GG TODO HERE: use haloes, eliminate all IFs, calculate phi + ext field on the fly without using phi2
  int i, j, k, l;
  int capped_flux = 0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1 ;j <= params.Ny ; j++ ){
	    int index = i * ( params.Ny + 2 ) + j;
	    for( k = 0; k < params.Nz; k++ ){
        /* Set starting flux at [index][k] = 0.0  */
        flux_site_plus[index][k]  = 0.0;
        flux_site_minus[index][k] = 0.0;
	      for( l = 0; l < 18; l++ ){  
	      /* Here I compute at once the flux in both directions. */
	        double jump_x = 0.0;;
	        double jump_y = 0.0;;
	        double jump_z = 0.0;;
	        int ip = i + x_dir[l];
	        if( ip > params.Nx ){
	          ip = 1;
	          jump_x = params.e_slope_x * params.Nx;
	        }
	        if( ip < 1 ){
	          ip = params.Nx;
	          jump_x = -params.e_slope_x * params.Nx;
	        }
	        int jp = j + y_dir[l];
	        if( jp > params.Ny ){
	          jp = 1;
	          jump_y = params.e_slope_y * params.Ny;
	        }
	        if( jp < 1 ){
	          jp = params.Ny;
	          jump_y = -params.e_slope_y * params.Ny;
	        }
	        int kp = k + z_dir[l];
	        if( kp > ( params.Nz - 1 )){
	          kp = 0;
	          jump_z = params.e_slope_z * params.Nz; 
	        }
	        if( kp < 0 ){
	          kp = params.Nz - 1;
	          jump_z = -params.e_slope_z * params.Nz; 
	        }
	        int indexp = ip * ( params.Ny + 2 ) + jp;

          if(inside[index][k]==FLUID){
	          if(inside[indexp][kp]==FLUID ){ /* both links in fluid */		
              real phi_tot    = phi_sor[index][k+1]   + ( params.e_slope_x * i + params.e_slope_y * j + params.e_slope_z * k );
              real phi_tot_nn = phi_sor[indexp][kp+1] + ( params.e_slope_x * ip + params.e_slope_y * jp + params.e_slope_z * kp );
              double exp_dphi, exp_min_dphi;
	            exp_dphi = exp( phi_tot_nn - phi_tot + jump_x + jump_y + jump_z );
	            exp_min_dphi = 1.0 / exp_dphi;
	            /* Flux due to electrostatic and density gradients */
              double flux_link_plus, flux_link_minus;
	            flux_link_plus =  -0.5 * ( 1 + exp_min_dphi ) * ( c_plus[indexp][kp] * exp_dphi - c_plus[index][k] );
	            flux_link_minus = -0.5 * ( 1 + exp_dphi ) * ( c_minus[indexp][kp] * exp_min_dphi - c_minus[index][k] );
              if( exp_min_dphi < 10e-6 || exp_dphi > 10e+6 ){ 
                printf( "\n@ %d %d %d link %d delta pot is %e\n", i, j, k, l, phi_tot_nn - phi_tot );
                printf("exp_MIN_dphi Way TOO small %e OR exp_dphi Way too big %e\n", exp_min_dphi, exp_dphi);
                printf("flux_+ %e flux_- %e \n", flux_link_plus, flux_link_minus );
              }	            flux_link_plus  *= ( D_plus  / delta[l] );
	            flux_link_minus *= ( D_minus / delta[l] );
              if( flux_link_plus > 1.0 ){
                //printf("F_l_+ > 1.0. Limiting to 1.0\n");
                //flux_link_plus = 0.01;
                capped_flux++;
              }
              if( flux_link_plus < -1.0 ){
                //printf("F_l_+ < -1.0. Limiting to -1.0\n");
                //flux_link_plus = -0.01;
                capped_flux++;
              }
              if( flux_link_minus > 1.0 ){
                //printf("F_l_- > 1.0. Limiting to 1.0\n");
                //flux_link_minus = 0.01;
                capped_flux++;
              }
              if( flux_link_minus < -1.0 ){
                //printf("F_l_- < -1.0. Limiting to -1.0\n");
                //flux_link_minus = -0.01;
                capped_flux++;
              }

              flux_site_plus[index][k]  += flux_link_plus / scale;
              flux_site_minus[index][k]  += flux_link_minus / scale;
	          }
          }
        }
        /* Midpoint 2nd order Runge-Kutta requires scale = 2.0 */
        flux_site_plus[index][k]  /= scale;
        flux_site_minus[index][k] /= scale;
        //printf( "%d %d %d plus %e minus %e xlf\n", i, j, k, flux_site_plus[index][k], flux_site_minus[index][k] );
      }
    }
  }
  if( capped_flux > 0 ) printf( "Capped fluxes %d times!!!\n", capped_flux );
}

void calc_fluxes2( double ** phi_sor, double ** c_plus, double ** c_minus, double ** flux_site_plus, double ** flux_site_minus, int ** inside, real *** solute_force, real scale )
{
  double D_plus, D_minus;
  D_plus  = params.D_plus  / params.multisteps;
  D_minus = params.D_minus / params.multisteps;
  
  // GG TODO HERE: use haloes, eliminate all IFs, calculate phi + ext field on the fly without using phi2
  int i, j, k, l;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1 ;j <= params.Ny ; j++ ){
	    int index = i * ( params.Ny + 2 ) + j;
	    for( k = 0; k < params.Nz; k++ ){
        /* Set starting flux at [index][k] = 0.0  */
        flux_site_plus[index][k]  = 0.0;
        flux_site_minus[index][k] = 0.0;
        /* Set starting solute_force = 0.0 */
        int s;
	      for( s = 0; s < 3; s++ ){
          solute_force[index][s][k] = 0.0;
        }
	      for( l = 0; l < 18; l++ ){  
	      /* Here I compute at once the flux in both directions. */
	        double jump_x = 0.0;;
	        double jump_y = 0.0;;
	        double jump_z = 0.0;;
	        int ip = i + x_dir[l];
	        if( ip > params.Nx ){
	          ip = 1;
	          jump_x = params.e_slope_x * params.Nx;
	        }
	        if( ip < 1 ){
	          ip = params.Nx;
	          jump_x = -params.e_slope_x * params.Nx;
	        }
	        int jp = j + y_dir[l];
	        if( jp > params.Ny ){
	          jp = 1;
	          jump_y = params.e_slope_y * params.Ny;
	        }
	        if( jp < 1 ){
	          jp = params.Ny;
	          jump_y = -params.e_slope_y * params.Ny;
	        }
	        int kp = k + z_dir[l];
	        if( kp > ( params.Nz - 1 )){
	          kp = 0;
	          jump_z = params.e_slope_z * params.Nz; 
	        }
	        if( kp < 0 ){
	          kp = params.Nz - 1;
	          jump_z = -params.e_slope_z * params.Nz; 
	        }
	        int indexp = ip * ( params.Ny + 2 ) + jp;

            real phi_tot    = phi_sor[index][k+1]   + ( params.e_slope_x * i + params.e_slope_y * j + params.e_slope_z * k );
            real phi_tot_nn = phi_sor[indexp][kp+1] + ( params.e_slope_x * ip + params.e_slope_y * jp + params.e_slope_z * kp );
            double exp_dphi, exp_min_dphi;
	          exp_dphi = exp( phi_tot_nn - phi_tot + jump_x + jump_y + jump_z );
	          exp_min_dphi = 1.0 / exp_dphi;
	          /* Flux due to electrostatic and density gradients */
            double flux_link_plus, flux_link_minus;
	          flux_link_plus =  -0.5 * ( 1 + exp_min_dphi ) * ( c_plus[indexp][kp] * exp_dphi - c_plus[index][k] );
	          flux_link_minus = -0.5 * ( 1 + exp_dphi ) * ( c_minus[indexp][kp] * exp_min_dphi - c_minus[index][k] );
	          real f_microions = params.kbt * ( flux_link_plus + flux_link_minus );
          if(inside[index][k]==FLUID){
	          if(inside[indexp][kp]==FLUID ){ /* both links in fluid */		
              flux_link_plus  *= ( D_plus  / delta[l] );
	            flux_link_minus *= ( D_minus / delta[l] );

              flux_site_plus[index][k]  += flux_link_plus / scale;
              flux_site_minus[index][k]  += flux_link_minus / scale;
              /* We exploit formula (22) JCP Fabrizio to calculate solute_force too */
              solute_force[index][0][k] += a1[l] * x_dir[l] * f_microions;
		          solute_force[index][1][k] += a1[l] * y_dir[l] * f_microions;
          		solute_force[index][2][k] += a1[l] * z_dir[l] * f_microions;
            /* If it is FLUID boundary node, there is no diffusion, but we need solute_force */
	          }else if( inside[indexp][kp] >= SOLID  ){
              solute_force[index][0][k] += a1[l] * x_dir[l] * f_microions;
		          solute_force[index][1][k] += a1[l] * y_dir[l] * f_microions;
              solute_force[index][2][k] += a1[l] * z_dir[l] * f_microions;
              /* This way (as in Fabrizio's code)  works ok, but if I introduce field would not work as fluid on boundary nodes does not feel charges 
               * IN ADDITION NO MOMENTUM CONSERVATION!!; 
              solute_force[index][0][k] = 0.0;
              solute_force[index][1][k] = 0.0;
              solute_force[index][2][k] = 0.0;
              */
            }
          }else{
            /* If it is SOLID or WALL, for now forces are calculated with gradient. GG TODO can I use LF method here too? */
            solute_force[index][0][k] = 0.0;
            solute_force[index][1][k] = 0.0;
            solute_force[index][2][k] = 0.0;
            /* This way on each node we calculate solute_force 
            solute_force[index][0][k] += a1[l] * x_dir[l] * f_microions;
		        solute_force[index][1][k] += a1[l] * y_dir[l] * f_microions;
            solute_force[index][2][k] += a1[l] * z_dir[l] * f_microions;
            */
          }
        }
        /* Midpoint 2nd order Runge-Kutta requires scale = 2.0 */
        flux_site_plus[index][k]  /= scale;
        flux_site_minus[index][k] /= scale;
        //printf( "%d %d %d plus %e minus %e xlf\n", i, j, k, flux_site_plus[index][k], flux_site_minus[index][k] );
      }
    }
  }
}

int move_charges( double ** c_plus, double ** c_minus, double ** flux_site_plus, double ** flux_site_minus, int ** inside )
{
  /* Update concentrations. Smoluchowski part */
  real tot_plus  = 0.0;
  real tot_minus = 0.0;
  real  maxflux, maxflux_p, maxflux_m;
  maxflux = maxflux_p = maxflux_m  = 0.0;
  int flag = 1;
  int i, j, k;
  int ap, bp, cp;
  int am, bm, cm;
  ap = bp = cp = am = bm = cm = 0;
  real max_accuracy_cp = 0.0;
  real max_accuracy_cm = 0.0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = i * ( params.Ny + 2 ) + j;
      for( k = 0; k < params.Nz; k++ ){
        if( inside[index][k] == FLUID ){
          c_plus [index][k] -= flux_site_plus[index][k];
          c_minus[index][k] -= flux_site_minus[index][k];
#ifdef WARREN_TEST
          c_minus[index][k]=0.0;
#endif
          real mf_try = fabs(flux_site_plus[index][k]) > fabs(flux_site_minus[index][k]) ? fabs(flux_site_plus[index][k]) : fabs(flux_site_minus[index][k]);
          if( mf_try > maxflux) {
            maxflux = mf_try;
            maxflux_p = fabs(flux_site_plus[index][k]);
            maxflux_m = fabs(flux_site_minus[index][k]);
          }

          real accuracy_cp = fabs( flux_site_plus[index][k]  / c_plus [index][k] );
          if( max_accuracy_cp < accuracy_cp ){
            max_accuracy_cp = accuracy_cp;
            ap = i; bp = j; cp = k;
          }
          real accuracy_cm = fabs( flux_site_minus[index][k] / c_minus [index][k] );
          if( max_accuracy_cm < accuracy_cm ){
            max_accuracy_cm = accuracy_cm;
            am = i; bm = j; cm = k;
          }

          if( max_accuracy_cp > params.diffusion_acc ||  max_accuracy_cm > params.diffusion_acc ){
            flag = 0;
          }
          tot_plus  += flux_site_plus[index][k];
          tot_minus += flux_site_minus[index][k];
        }
      }   
    }
  }
  if( !flag ){
    printf("WARNING: SMALL ACCURACY plus %e (%d %d %d) minus %e (%d %d %d) MAXFLUX %e (F+ %e F- %e)\n", 
      max_accuracy_cp, ap, bp, cp, max_accuracy_cm, am, bm, cm, maxflux, maxflux_p, maxflux_m );
    if( (max_accuracy_cp > params.max_eq_fluxes ||  max_accuracy_cm > params.max_eq_fluxes) && !strcmp(params.restart, "no")  ){
      params.D_therm /= 2.0; params.D_plus /= 2.0; params.D_minus /= 2.0;
      printf( "\nALARM: Charges are moving about quite quickly! Setting diffusions to %e %e\n", params.D_plus, params.D_minus );
    }
  }
  if( (fabs(tot_plus) > 1e-12) || (fabs(tot_minus) > 1e-12) ){
    printf("BIG TROUBLES!\n");
    printf("The sum of all fluxes does not add up to ZERO!\n");
    printf("Tot_plus_fluxes = %e\n",tot_plus);
    printf("Tot_minus_fluxes = %e\n",tot_minus);
    exit(5);
  }
  return flag;
}

int diffuse_euler( double ** c_plus, double ** c_minus, double ** phi_sor, int ** inside )
{
  int max_index = ( params.Nx + 2 ) * ( params.Ny + 2 ); 
  real ** flux_site_plus   = mem_r_lattice2( max_index, params.Nz );
  real ** flux_site_minus  = mem_r_lattice2( max_index, params.Nz );

  // Assuming potential and charges in sync 
  // Calc fluxes
  calc_fluxes( phi_sor, c_plus, c_minus, flux_site_plus, flux_site_minus, inside, 1.0 );
  // Update charge distributions
  int equilibrated = move_charges( c_plus, c_minus, flux_site_plus, flux_site_minus, inside );
  // Now run sor to keep in sync charges and potential at t + dt
  sor2( params.Nx, params.Ny, params.Nz, c_minus, c_plus, phi_sor );

  free_2D( (void **) flux_site_plus );
  free_2D( (void **) flux_site_minus );
  return equilibrated;
}

int diffuse_euler2( double ** c_plus, double ** c_minus, double ** phi_sor, int ** inside, real *** solute_force )
{
  int max_index = ( params.Nx + 2 ) * ( params.Ny + 2 ); 
  real ** flux_site_plus   = mem_r_lattice2( max_index, params.Nz );
  real ** flux_site_minus  = mem_r_lattice2( max_index, params.Nz );

  // Assuming potential and charges in sync 
  // Calc fluxes
  //calc_fluxes( phi_sor, c_plus, c_minus, flux_site_plus, flux_site_minus, inside, 1.0 );
  calc_fluxes2( phi_sor, c_plus, c_minus, flux_site_plus, flux_site_minus, inside, solute_force, 1.0 );
  // Update charge distributions
  int equilibrated = move_charges( c_plus, c_minus, flux_site_plus, flux_site_minus, inside );
  // Now run sor to keep in sync charges and potential at t + dt
  sor2( params.Nx, params.Ny, params.Nz, c_minus, c_plus, phi_sor );

  free_2D( (void **) flux_site_plus );
  free_2D( (void **) flux_site_minus );
  return equilibrated;
}
void move_d_charges( double ** c_plus, double ** c_minus, double ** flux_site_plus, double ** flux_site_minus, int ** inside, int scale )
{
  /* Update concentrations. Smoluchowski part */
  real tot_plus  = 0.0;
  real tot_minus = 0.0;
  real  maxflux, maxflux_p, maxflux_m;
  maxflux = maxflux_p = maxflux_m  = 0.0;
  int flag_tolerance = 0;
  int i, j, k;
  int a, b, c;
  a = b = c = 0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = i * ( params.Ny + 2 ) + j;
      for( k = 0; k < params.Nz; k++ ){
        if( inside[index][k] == FLUID ){
          c_plus [index][k] -= flux_site_plus[index][k];
          c_minus[index][k] -= flux_site_minus[index][k];
#ifdef WARREN_TEST
           c_minus[index][k]=0.0;
#endif
          if( fabs(flux_site_plus[index][k]) > params.diffusion_acc || fabs(flux_site_minus[index][k]) > params.diffusion_acc ){
            flag_tolerance = 1;
            real mf_try = fabs(flux_site_plus[index][k]) > fabs(flux_site_minus[index][k]) ? fabs(flux_site_plus[index][k]) : fabs(flux_site_minus[index][k]);
            if( mf_try > maxflux) {
              maxflux = mf_try;
              maxflux_p = fabs(flux_site_plus[index][k]);
              maxflux_m = fabs(flux_site_minus[index][k]);
              a = i; b = j; c = k;
            }
          }
          tot_plus  += flux_site_plus[index][k] * scale;
          tot_minus += flux_site_minus[index][k] * scale;
        }
      }   
    }
  }
  if( flag_tolerance ) printf("WARNING: MAXFLUX %e (F+ %e F- %e) in %d %d %d\n", maxflux, maxflux_p, maxflux_m, a, b, c);
  if( (fabs(tot_plus) > 1e-12) || (fabs(tot_minus) > 1e-12) ){
    printf("BIG TROUBLES!\n");
    printf("The sum of all fluxes does not add up to ZERO!\n");
    printf("Tot_plus_fluxes = %e\n",tot_plus);
    printf("Tot_minus_fluxes = %e\n",tot_minus);
    exit(5);
  }
}

void reshape3D( real ** l_old, real ** l_new )
{
  int x, y, index;
  /* memcpy l_old into the inner part of l_new */
  for( x = 1; x <= params.Nx; x++){
    for( y = 1; y <= params.Ny; y++){
      index = x * ( params.Ny + 2 ) + y;
      int columns_offset = x * ( params.Ny + 2 ) + y;
      int sites_offset = columns_offset * ( params.Nz + 2 ) + 1;
      memcpy( l_new[0] + sites_offset, l_old[index], params.Nz * sizeof(real) );
    }
  }
  /* memcpy PBC for XY faces l_new */
  for( y = 1; y <= params.Ny; y++){
    index = 1 * ( params.Ny + 2 ) + y;
    int columns_offset = ( params.Nx + 1 ) * ( params.Ny + 2 ) + y;
    int sites_offset = columns_offset * ( params.Nz + 2 ) + 1;
    memcpy( l_new[0] + sites_offset, l_old[index], params.Nz * sizeof(real) );

    index = params.Nx * ( params.Ny + 2 ) + y;
    columns_offset = 0 * ( params.Ny + 2 ) + y;
    sites_offset = columns_offset * ( params.Nz + 2 ) + 1;
    memcpy( l_new[0] + sites_offset, l_old[index], params.Nz * sizeof(real) );
  }
  for( x = 1; x <= params.Nx; x++){
    index = x * ( params.Ny + 2 ) + 1;
    int columns_offset = x * ( params.Ny + 2 ) + 1 + params.Ny;
    int sites_offset = columns_offset * ( params.Nz + 2 ) + 1;
    memcpy( l_new[0] + sites_offset, l_old[index], params.Nz * sizeof(real) );

    index = x * ( params.Ny + 2 ) + params.Ny;
    columns_offset = x * ( params.Ny + 2 );
    sites_offset = columns_offset * ( params.Nz + 2 ) + 1;
    memcpy( l_new[0] + sites_offset, l_old[index], params.Nz * sizeof(real) );
  }
  /* memcpy PBC for 4 XY corners */
  int inx_c1, inx_c2, inx_c3, inx_c4;
  inx_c1 = ( params.Ny + 2 ) + 1; 
  inx_c2 = ( params.Ny + 2 ) + 1 + params.Ny; 
  inx_c3 = params.Nx * ( params.Ny + 2 ) + 1; 
  inx_c4 = params.Nx * ( params.Ny + 2 ) + 1 + params.Ny; 
  int columns_offset = ( params.Nx + 1 ) * ( params.Ny + 2 ) + 1 + params.Ny;
  int sites_offset = columns_offset * ( params.Nz + 2 ) + 1;
  memcpy( l_new[0] + sites_offset, l_old[inx_c1], params.Nz * sizeof(real) );
  columns_offset = ( params.Nx + 1 ) * ( params.Ny + 2 );
  memcpy( l_new[0] + sites_offset, l_old[inx_c2], params.Nz * sizeof(real) );
  columns_offset = 1 + params.Ny;
  memcpy( l_new[0] + sites_offset, l_old[inx_c3], params.Nz * sizeof(real) );
  columns_offset = 0;
  memcpy( l_new[0] + sites_offset, l_old[inx_c4], params.Nz * sizeof(real) );
  
  /* Z-PBC for each XY */
  for( x = 1; x <= params.Nx; x++){
    for( y = 1; y <= params.Ny; y++){
      index = x * ( params.Ny + 2 ) + y;
      l_new[index][0] = l_new[index][params.Nz - 1];
      l_new[index][params.Nz] = l_new[index][1];
    }
  }
}

void sor2( int lx, int ly, int lz, double **c_minus, double **c_plus, double **phi )
{
  int maxits=50000;
  extern real anormf0;
  double anorm, anormmax, anormf,omega,resid;
  double factor;
  double phistar, phiold, rjac2;
  int ipass,isw,jsw,ksw,index,index1,index2;
  int isw2,jsw2,ksw2;
  int i, j, k, imin, jmin, kmin, n;
  int veldir;
  int finished=0;
  
  int siz_y = params.Ny + 2;
  real ** phitmp = mem_r_lattice2( (params.Nx + 2) * (params.Ny + 2), params.Nz + 2 ); 

  rjac2=4.0/7.0;
  omega = 1.43;
  factor= PIG * params.ch_bjl;

  /* anormf controls sor precision (see below) */
  anormf = 0.0;
  for(i=1;i<=lx;i++){
    for(j=1;j<=ly;j++){
      index=i*siz_y+j;
      for(k=1;k<=lz;k++){
	      anormf+= fabs( phi[index][k] );
      }  
    } 
  }  
  /* This happens at firs sor, when phi = 0.0. anormf0 calc @ init, sum |charges| * factor */
  if( anormf == 0.0 ) anormf = anormf0;

  for( n = 1; n <= maxits; n++ ){
    anormmax = 0.0;
    anorm=0.0;
    ksw  =1;
    ksw2 =1;
    for( ipass = 1; ipass <= 2; ipass++ ){
      /* Calc ingredients for new value of phi */
      jsw=ksw;
      for( k = 1; k <= lz; k++ ){
	      isw = jsw;
	      for( j = 1; j <= ly; j++ ){
	        for( i = isw; i <= lx; i += 2 ){
	          index1 = i * siz_y + j;	    
	          phistar = 0.0;
	          for( veldir = 0; veldir < 18; veldir++ ){
	            imin = i - x_dir[veldir];
	            jmin = j - y_dir[veldir];
	            index2 = imin * siz_y + jmin;
	            kmin = k - z_dir[veldir];
	            phistar += a0[veldir] * phi[index2][kmin];
	          }
            /* GG Here k - 1 as c_\pm still have an old lattice shape format */
	          phistar += factor * ( c_plus[index1][k-1] - c_minus[index1][k-1] );
	          phiold = phi[index1][k];
	          phitmp[index1][k] = omega * phistar + ( 1.0 - omega ) * phiold;
	          resid = phistar - phiold;
	          anorm += fabs( resid );
	        }
	        isw = 3 - isw;
	      }
	      jsw = 3 - jsw;
      }
      ksw = 3 - ksw;

      /* Assign new value to phi */
      jsw2 = ksw2;
      for( k = 1; k <= lz; k++ ){
	      isw2 = jsw2;
	      for(j = 1; j <= ly; j++ ){
	        for( i = isw2; i <= lx; i += 2 ){
	          index = i * siz_y + j;
	          phi[index][k] = phitmp[index][k];
	        }
	        isw2 = 3 - isw2;
	      }
	      jsw2 = 3 - jsw2;
      }
      ksw2 = 3 - ksw2;

      /* Finished pass (i.e. calc of new phi values on alternate half of lattice points) 
       * as we do not use if, updated values of phis in halo regions are required */
      sorPBC( phi );
    }  
    sorPBC( phi );

    real convergence = fabs( params.poisson_acc * anormf );
    if( anorm <= convergence ){
      finished=1;
      free_2D( (void **) phitmp );
      if( n != 1 ) printf("Sor2 was run %d times before reaching tolerance (modulo anormf) of %f\n", 
      n, params.poisson_acc );
      break;
    }    
  } 

  if( finished ==0 ){
    printf("maxits exceeded in sor");
    exit(1);
  }  
}

void sorPBC( real ** phi )
{
  int x, y, index;
  /* memcpy PBC for XY faces phi */
  for( y = 1; y <= params.Ny; y++){
    index = 1 * ( params.Ny + 2 ) + y;
    int columns_offset = ( params.Nx + 1 ) * ( params.Ny + 2 ) + y;
    int sites_offset = columns_offset * ( params.Nz + 2 ) + 1;
    /* Note phi[index] + 1 */
    memcpy( phi[0] + sites_offset, phi[index] + 1, params.Nz * sizeof(real) );

    index = params.Nx * ( params.Ny + 2 ) + y;
    columns_offset = 0 * ( params.Ny + 2 ) + y;
    sites_offset = columns_offset * ( params.Nz + 2 ) + 1;
    /* Note phi[index] + 1 */
    memcpy( phi[0] + sites_offset, phi[index] + 1, params.Nz * sizeof(real) );
  }
  for( x = 1; x <= params.Nx; x++){
    index = x * ( params.Ny + 2 ) + 1;
    int columns_offset = x * ( params.Ny + 2 ) + 1 + params.Ny;
    int sites_offset = columns_offset * ( params.Nz + 2 ) + 1;
    /* Note phi[index] + 1 */
    memcpy( phi[0] + sites_offset, phi[index] + 1, params.Nz * sizeof(real) );

    index = x * ( params.Ny + 2 ) + params.Ny;
    columns_offset = x * ( params.Ny + 2 );
    sites_offset = columns_offset * ( params.Nz + 2 ) + 1;
    /* Note phi[index] + 1 */
    memcpy( phi[0] + sites_offset, phi[index] + 1, params.Nz * sizeof(real) );
  }
  /* memcpy PBC for 4 XY corners */
  int inx_c1, inx_c2, inx_c3, inx_c4;
  inx_c1 = ( params.Ny + 2 ) + 1; 
  inx_c2 = ( params.Ny + 2 ) + params.Ny; 
  inx_c3 = params.Nx * ( params.Ny + 2 ) + 1; 
  inx_c4 = params.Nx * ( params.Ny + 2 ) + params.Ny; 
  /* c1 -> halo c4 */
  int columns_offset = ( params.Nx + 1 ) * ( params.Ny + 2 ) + 1 + params.Ny;
  int sites_offset = columns_offset * ( params.Nz + 2 ) + 1;
  memcpy( phi[0] + sites_offset, phi[inx_c1] + 1, params.Nz * sizeof(real) );
  /* c2 -> halo c3 */
  columns_offset = ( params.Nx + 1 ) * ( params.Ny + 2 );
  sites_offset = columns_offset * ( params.Nz + 2 ) + 1;
  memcpy( phi[0] + sites_offset, phi[inx_c2] + 1, params.Nz * sizeof(real) );
  /* c3 -> halo c2 */
  columns_offset = 1 + params.Ny;
  sites_offset = columns_offset * ( params.Nz + 2 ) + 1;
  memcpy( phi[0] + sites_offset, phi[inx_c3] + 1, params.Nz * sizeof(real) );
  /* c4 -> halo c1 */
  columns_offset = 0;
  sites_offset = columns_offset * ( params.Nz + 2 ) + 1;
  memcpy( phi[0] + sites_offset, phi[inx_c4] + 1, params.Nz * sizeof(real) );
  
  /* Z-PBC for each XY */
  for( x = 0; x <= params.Nx + 1; x++){
    for( y = 0; y <= params.Ny + 1; y++){
      index = x * ( params.Ny + 2 ) + y;
      phi[index][0] = phi[index][params.Nz];
      phi[index][params.Nz + 1] = phi[index][1];
    }
  }
}

void sor( int lx, int ly, int lz, double **c_minus, double **c_plus, double **phi )
{
  int maxits=50000;
  extern real anormf0;
  double anorm, anormmax, anormf,omega,resid;
  double factor;
  double phistar, phiold, rjac2;
  /* C declared variables */
  int ipass,isw,jsw,ksw,index,index1,index2;
  int isw2,jsw2,ksw2;
  int i,j,k,imin,jmin,kmin,n;
  int veldir;
  int finished=0;
  double tmpvar_phi;
  int siz_y = params.Ny + 2;
  
  int max_index = ( params.Nx + 2 ) * ( params.Ny + 2 );
  double ** phitmp   = mem_r_lattice2( max_index, params.Nz );


  rjac2=4.0/7.0;
  omega = 1.43;
  factor= PIG * params.ch_bjl;
  tmpvar_phi=0.0;

  /* anormf controls sor precision (see below) */
  anormf = 0.0;
  for(i=1;i<=lx;i++){
    for(j=1;j<=ly;j++){
      index=i*siz_y+j;
      for(k=0;k<lz;k++){
	      anormf+= fabs( phi[index][k] );
	      tmpvar_phi += phi[index][k];
      }  
    } 
  }  
  /* This happens at firs sor, when phi = 0.0. anormf0 calc @ init, sum |charges| * factor */
  if( anormf == 0.0 ) anormf = anormf0;

  for( n = 1; n <= maxits; n++ ){
    anormmax = 0.0;
    anorm=0.0;
    ksw  =1;
    ksw2 =1;
    for( ipass = 1; ipass <= 2; ipass++ ){
      jsw=ksw;
      for( k = 0; k < lz; k++ ){
	      isw = jsw;
	      for( j = 1; j <= ly; j++ ){
	        for( i = isw; i <= lx; i += 2 ){
	          index1 = i * siz_y + j;	    
	          phistar = 0.0;
	          for( veldir = 0; veldir < 18; veldir++ ){
	            imin = i - x_dir[veldir];
	            /* PBC */
	            if( imin > lx ) imin = 1;
	            if( imin < 1  ) imin = lx;
	            /* EOF PBC */
	            jmin = j - y_dir[veldir];
	            /* PBC */
	            if( jmin > ly) jmin = 1;
	            if( jmin < 1 ) jmin = ly;
	            /* EOF PBC */
	            /* NO PBC*/
	            //if(jmin>ly)jmin=ly;
	            //if(jmin<1) jmin = 1;
	            /*EOF NO PBC*/
	            index2 = imin * siz_y + jmin;
	            kmin = k - z_dir[veldir];
	            /* PBC */
	            if( params.width < 1.0 ){
		          if( kmin > lz - 1 ) kmin = 0;
		          if( kmin < 0 ) kmin = lz - 1;
	            }else{
		          if( kmin > lz - 1 ) kmin = lz - 1;
		          if( kmin < 0 ) kmin = 0;
	            }
	            /* EOF PBC */
	            /*    imin=plusx[ i-x_dir[veldir] ];
		          jmin=plusy[ j-y_dir[veldir] ];
		          kmin=plusz[ k-z_dir[veldir] ];
		          index=imin*siz_y+jmin;*/
	            phistar += a0[veldir] * phi[index2][kmin];
	          }
	          phistar += factor * ( c_plus[index1][k] - c_minus[index1][k] );
	          phiold = phi[index1][k];
	          phitmp[index1][k] = omega * phistar + ( 1.0 - omega ) * phiold;
	          //printf("phistar %e phiold %e phitmp %e %e at %d\n",phistar,phiold,c_plus[index1][k],c_minus[index1][k],i);
	          resid = phistar - phiold;
	          anorm += fabs( resid );
            //anormmax=max(anormmax,fabs(resid)); 
	        }
	        isw = 3 - isw;
	      }
	      jsw = 3 - jsw;
      }
      ksw = 3 - ksw;
      jsw2 = ksw2;
      for( k = 0; k < lz; k++ ){
	      isw2 = jsw2;
	      for(j = 1; j <= ly; j++ ){
	        for( i = isw2; i <= lx; i += 2 ){
	          index = i * siz_y + j;
	          phi[index][k] = phitmp[index][k];
	        }
	        isw2 = 3 - isw2;
	      }
	      jsw2 = 3 - jsw2;
      }
      ksw2 = 3 - ksw2;
      //omega=(n==1&&ipass==1?1.0/(1.0-0.5*rjac2):1.0/(1.0-0.25*rjac2*omega));
    }  
    tmpvar_phi = 0.0;
    for( i = 1; i <= lx; i++ ){
      for( j = 1; j <= ly; j++ ){
	      index = i * siz_y + j;
	      for( k = 0; k < lz; k++ ){
	      tmpvar_phi += phi[index][k];
	      }  
      }   
    }    
    /*       Write a sample of the iterations */    
    if( n % 5000 == 0 ) printf("Sor hit %d\n", n);

    if( anorm <= params.poisson_acc * anormf ){
      finished=1;
      if( n != 1 ) printf("Sor was run %d times before reaching tolerance (modulo anormf) of %f\n", n, params.poisson_acc );
      break;
    }    
  } 
  if( finished ==0 ){
    printf("maxits exceeded in sor");
    exit(1);
  }  
  free_2D( (void **) phitmp );
}

void calc_gradient( real **phi_sor, double ** phi_p_x, double ** phi_p_y, double ** phi_p_z )
{
  int siz_y = params.Ny + 2;
  real d_Ex = params.e_slope_x;
  real d_Ey = params.e_slope_y;
  real d_Ez = params.e_slope_z;
  int i, j, k, l;

  for( i = 1; i <= params.Nx; i++){
    for(j = 1; j <= params.Ny; j++){
      int index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        double grad_phi_x = 0.0;
        double grad_phi_y = 0.0;
        double grad_phi_z = 0.0;
        /* See how jump is addded ( jump += ... not = as before, wrong!) */
        for( l = 0; l < 18; l++ ){  
          double jump_x = 0.0;
          double jump_y = 0.0;
          double jump_z = 0.0;
          int index_nn;
          int i_nn = i + x_dir[l];
          if( i_nn > params.Nx ){
            i_nn = 1;
            jump_x += d_Ex * params.Nx;
          }
          if( i_nn < 1 ){
            i_nn = params.Nx;
            jump_x += -d_Ex * params.Nx;
          }
          int j_nn = j + y_dir[l];
          if( j_nn > params.Ny ){
            j_nn = 1;
            jump_y += d_Ey * params.Ny;
          }
          if( j_nn < 1 ){
            j_nn = params.Ny;
            jump_y += -d_Ey * params.Ny;
          }
          int k_nn = k + z_dir[l];
          if( k_nn > (params.Nz - 1) ){
            k_nn = 0;
            jump_z += d_Ez * params.Nz;
          }
          if( k_nn < 0 ){
            k_nn = ( params.Nz - 1 );
            jump_z += -d_Ez * params.Nz;
          }

          index_nn = i_nn * siz_y + j_nn;
          grad_phi_x += a1[l] * x_dir[l] * ( phi_sor[index_nn][k_nn+1] + d_Ex * i_nn + d_Ey * j_nn + d_Ez * k_nn + jump_x + jump_y + jump_z );
          grad_phi_y += a1[l] * y_dir[l] * ( phi_sor[index_nn][k_nn+1] + d_Ex * i_nn + d_Ey * j_nn + d_Ez * k_nn + jump_x + jump_y + jump_z );
          grad_phi_z += a1[l] * z_dir[l] * ( phi_sor[index_nn][k_nn+1] + d_Ex * i_nn + d_Ey * j_nn + d_Ez * k_nn + jump_x + jump_y + jump_z );
          /* THIS TO LEAVE ONLY EXTERNAL FIELD, TO SEE FIRST TIMESTEP IF GRADIENT CALC IS OK AND SOLUTE FORCE OK
          grad_phi_x += a1[l] * x_dir[l] * ( d_Ex * i_nn + d_Ey * j_nn + d_Ez * k_nn + jump );
          grad_phi_y += a1[l] * y_dir[l] * ( d_Ex * i_nn + d_Ey * j_nn + d_Ez * k_nn + jump );
          grad_phi_z += a1[l] * z_dir[l] * ( d_Ex * i_nn + d_Ey * j_nn + d_Ez * k_nn + jump );
          */
        }
        phi_p_x[index][k+1] = grad_phi_x;
        phi_p_y[index][k+1] = grad_phi_y;
        phi_p_z[index][k+1] = grad_phi_z;
      } 
    }
  }
}

void f_el_ext_colloids( object * obj, real **c_plus, real **c_minus, int **inside, double ** phi_p_x, double ** phi_p_y, double ** phi_p_z, real * tot_c_mom )
{
  int i, j, k, o;
  int siz_y = params.Ny + 2;

  tot_c_mom[0] = tot_c_mom[1] = tot_c_mom[2] = 0.0;
  for( o = 0; o < params.num_obj; o++){
    obj[o].f_ext.x = 0.0;
    obj[o].f_ext.y = 0.0;
    obj[o].f_ext.z = 0.0;

    // GG Case with two particles does not work if colloids are moving (see init.c:bnodes_init,497)
    if( params.num_obj == 2 ){
      int l_x = obj[o].i.x + obj[o].size_m.x;
      int l_y = obj[o].i.y + obj[o].size_m.y;
      int l_z = obj[o].i.z + obj[o].size_m.z;
      int r_x = obj[o].i.x + obj[o].size_p.x;
      int r_y = obj[o].i.y + obj[o].size_p.y;
      int r_z = obj[o].i.z + obj[o].size_p.z;
      for( i = l_x; i <= r_x; i++){
        for( j = l_y; j <= r_y; j++){
          int index = i * siz_y + j;
          for( k = l_z; k < r_z; k++ ){
            if( inside[index][k] >= SOLID ){
              obj[o].f_ext.x += - params.kbt * phi_p_x[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
              obj[o].f_ext.y += - params.kbt * phi_p_y[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
              obj[o].f_ext.z += - params.kbt * phi_p_z[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
              tot_c_mom[0]   += - params.kbt * phi_p_x[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
              tot_c_mom[1]   += - params.kbt * phi_p_y[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
              tot_c_mom[2]   += - params.kbt * phi_p_z[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
            }
          }
        }
      }
    /* In case of one colloid, it is moving,but nodes are not, so cannot use l_x, r_x, etc 
     * but as it is only one colloid I can loop over the whole lattice */
    }else if( params.num_obj == 1 ){ 
      for( i = 1; i <= params.Nx; i++){
        for( j = 1; j <= params.Ny; j++){
          int index = i * siz_y + j;
          for( k = 0; k < params.Nz; k++ ){
            if( inside[index][k] >= SOLID ){
              obj[o].f_ext.x += - params.kbt * phi_p_x[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
              obj[o].f_ext.y += - params.kbt * phi_p_y[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
              obj[o].f_ext.z += - params.kbt * phi_p_z[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
              tot_c_mom[0]   += - params.kbt * phi_p_x[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
              tot_c_mom[1]   += - params.kbt * phi_p_y[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
              tot_c_mom[2]   += - params.kbt * phi_p_z[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
            }
          }
        }
      }
    }
  }

  /* In this case there are many parts and you have to understand to which COLLOID each SOLID element belongs to */
  if( params.num_obj > 2 ){ 
    /* Resetting all forces on colloids to zero */
    for( o = 0; o < params.num_obj; o++){
      obj[o].f_ext.x = 0.0;
      obj[o].f_ext.y = 0.0;
      obj[o].f_ext.z = 0.0;
    }
    for( i = 1; i <= params.Nx; i++){
      for( j = 1; j <= params.Ny; j++){
        int index = i * siz_y + j;
        for( k = 0; k < params.Nz; k++ ){
          if( inside[index][k] >= SOLID ){
            int this_obj = inside[index][k] - SOLID;
            obj[this_obj].f_ext.x += - params.kbt * phi_p_x[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
            obj[this_obj].f_ext.y += - params.kbt * phi_p_y[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
            obj[this_obj].f_ext.z += - params.kbt * phi_p_z[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
            tot_c_mom[0]   += - params.kbt * phi_p_x[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
            tot_c_mom[1]   += - params.kbt * phi_p_y[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
            tot_c_mom[2]   += - params.kbt * phi_p_z[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
          }
        }
      }
    }
  }
  printf("\n");
  for( o = 0; o < params.num_obj; o++){
    printf("El. Momentum on colloid %d is %e %e %e\n", obj[o].self, obj[o].f_ext.x, obj[o].f_ext.y, obj[o].f_ext.z  );
  }
  printf("Total el momentum on colloids is %e %e %e\n", tot_c_mom[0], tot_c_mom[1], tot_c_mom[2] );
}

void f_el_ext_fluid( real **c_plus, real **c_minus, int **inside, double ** phi_p_x, double ** phi_p_y, double ** phi_p_z, real * tot_f_mom, real *** solute_force )
{
  int i, j, k;
  int siz_y = params.Ny + 2;

  tot_f_mom[0] = tot_f_mom[1] = tot_f_mom[2] = 0.0;
  for( i = 1; i <= params.Nx; i++){
    for(j = 1; j <= params.Ny; j++){
      int index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        /* If there is more than one colloid the force on FLUID nodes must be calculated only once */
        if( inside[index][k] == FLUID ){
          solute_force[index][0][k] = - params.kbt * phi_p_x[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
          solute_force[index][1][k] = - params.kbt * phi_p_y[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
          solute_force[index][2][k] = - params.kbt * phi_p_z[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
          tot_f_mom[0] += - params.kbt * phi_p_x[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
          tot_f_mom[1] += - params.kbt * phi_p_y[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
          tot_f_mom[2] += - params.kbt * phi_p_z[index][k+1] * ( c_plus[index][k] - c_minus[index][k] );
        }else{
          /* This is needed in case colloids move. */
          solute_force[index][0][k] = 0.0;
          solute_force[index][1][k] = 0.0;
          solute_force[index][2][k] = 0.0;
        }
      }
    }
  }
}

void inj_el_momentum( int **inside, real * tot_c_mom, real * tot_f_mom, object * obj, real *** solute_force )
{
  int i, j, k;
  int siz_y = params.Ny + 2;
  static real tot_elec_mom = 0.0;

  vector mom2insert;
  mom2insert.x = -( tot_c_mom[0] + tot_f_mom[0] ); 
  tot_elec_mom += mom2insert.x;
  mom2insert.y = -( tot_c_mom[1] + tot_f_mom[1] ); 
  mom2insert.z = -( tot_c_mom[2] + tot_f_mom[2] ); 
  printf("Total elec momentum to inj in FLUID %e %e %e in COLLOID %e %e %e\nLEVELLING: Total elec mom_x to inj in FLUID is %e. Running sum of elec moms(x) injected is %e\n\n", tot_f_mom[0], 
      tot_f_mom[1], tot_f_mom[2], tot_c_mom[0], tot_c_mom[1], tot_c_mom[2], mom2insert.x, tot_elec_mom );
  /* Distribute mom2insert over F+C */
  mom2insert.x /= (real) ( fluid_nodes );
  mom2insert.y /= (real) ( fluid_nodes);
  mom2insert.z /= (real) ( fluid_nodes);
  for( i = 1; i <= params.Nx; i++){
    for(j = 1; j <= params.Ny; j++){
      int index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        /* Put all momentum in solvent */
        if( inside[index][k] == FLUID ){
          solute_force[index][0][k] += mom2insert.x;
          solute_force[index][1][k] += mom2insert.y;
          solute_force[index][2][k] += mom2insert.z;
        }
      }
    }    
  }
}

void calc_current( object * obj, real *** velcs_ptr, real **c_plus, real **c_minus, real * J )
{
  J[0] = J[1] = J[2] = 0.0;
  int i, j, k;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = i * ( params.Ny + 2 ) + j;
      for( k = 0; k < params.Nz; k++ ){
        float tot_mass, tot_px, tot_py, tot_pz;
        int velc_dir;
        tot_mass = tot_px = tot_py = tot_pz = 0.0;
        for( velc_dir = 0; velc_dir < 18; velc_dir++){
          tot_mass += velcs_ptr[index][velc_dir][k];
          tot_px   += velcs_ptr[index][velc_dir][k] * x_dir[velc_dir];
          tot_py   += velcs_ptr[index][velc_dir][k] * y_dir[velc_dir];
          tot_pz   += velcs_ptr[index][velc_dir][k] * z_dir[velc_dir];
        }
        J[0] += ( c_plus[index][k] - c_minus[index][k] ) * tot_px / tot_mass;
        J[1] += ( c_plus[index][k] - c_minus[index][k] ) * tot_py / tot_mass;
        J[2] += ( c_plus[index][k] - c_minus[index][k] ) * tot_pz / tot_mass;
      }
    }
  }
}
