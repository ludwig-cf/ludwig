#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "ch_dynamics.h"
#include "simpar.h"
#include "init.h"
#include "lb.h"

extern sim_params_t params;
extern long int idum;

void dh_potential( real ** dh, double zeta, double lambda_d, object * obj )
{
  double cx  = obj->r.x;
  double cy  = obj->r.y;
  double cz  = obj->r.z;
  double rad = obj->rad;
  int siz_y = params.Ny + 2;

  int i, j, k;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        real dist = pow( pow((i - cx), 2.0) + pow((j - cy), 2.0) + pow((k - cz),2.0), 0.5 ) - rad;
        if( dist < 0.0 ){ dh[index][k] = 0.0; continue; }
        dh[index][k] = zeta * rad / (rad + dist) * exp( -dist / lambda_d );
        //printf( "%d %d %d are %f far away from surface and dh is %f\n", i, j, k, dist, dh[s] );
      }
    }
  }
}

extern real anormf0;
void place_charges_exp( real ** c_plus, real ** c_minus, real ** dh, int ** site_type )
{
  int siz_y = params.Ny + 2;

  /* Normalization factor (only on FLUID nodes) */
  real norm_minus = 0.0;
  real norm_plus  = 0.0;
  int i, j, k;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        if( site_type[index][k] == FLUID ){
          norm_minus += exp( + dh[index][k] );
          norm_plus  += exp( - dh[index][k] );
        }
      }
    }
  }  
  printf( "Normalizing factors are (+, %e) (-, %e)\n", norm_plus, norm_minus );

  /* Count solid and fluid sites */
  int tot_sites, obj_sites, fluid_sites;
  tot_sites = 0; obj_sites = 0; fluid_sites = 0; 
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = siz_y * i + j;
      for( k = 0; k < params.Nz; k++ ){
        if( site_type[index][k] == SOLID ) obj_sites++;
        if( site_type[index][k] == FLUID ) fluid_sites++;
        tot_sites++;
      }
    }
  }
  /* Work out charge density per solid site */ 
  real objects_charge = params.ch_objects * params.num_obj;
  real rho_solid;
  real rho_p_solid = 0.0; 
  real rho_m_solid = 0.0;
  if( obj_sites ){
    rho_solid = objects_charge / (real) obj_sites;
    rho_p_solid = 0.5 * rho_solid; 
    rho_m_solid = -rho_p_solid;
  }
  /* Calculate rho_fluid and change sign to it to guarantee electroneutrality */
  real rho_fluid  = ( params.num_obj ) ? ( objects_charge ) : 0;
  rho_fluid = ( rho_fluid ) / (real) fluid_sites;
  rho_fluid = -rho_fluid;
  /* Add salt to fluid */
  real con_salt = 1.0 / ( 4 * PIG * params.ch_bjl * params.ch_lambda * params.ch_lambda );
  if( params.ch_lambda == 0.0 ){  
    con_salt = 0.0;
    printf("\t*****   THIS IS A NO SALT SIMULATION     *****\n");
  }

  /* Update fluid concentration as there is now salt */
  real rho_p_fluid, rho_m_fluid;
  if( rho_fluid >= 0.0){
    rho_p_fluid = rho_fluid + 0.5 * con_salt; 
    rho_m_fluid = 0.5 * con_salt; 
  }else{
    rho_m_fluid = 0.5 * con_salt - rho_fluid; 
    rho_p_fluid = 0.5 * con_salt; 
  }

  /* Distribute charge to OBJECTS */
  real placed_o_charge = 0.0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = siz_y * i + j;
      for( k = 0; k < params.Nz; k++ ){
        if( site_type[index][k] == SOLID  ){
  	      c_plus [index][k] = rho_p_solid;
	        c_minus[index][k] = rho_m_solid;
	        placed_o_charge += c_plus[index][k] - c_minus[index][k];
        }
      }
    }
  }
  printf( "Placed UNIFORMELY total %f charge on OBJECTS out of the %f(*%d) input.\nOBJECTS densities (+|-) are (%f|%f)\n", 
  placed_o_charge, params.ch_objects, params.num_obj, rho_p_solid, rho_m_solid );

  /* Work out how much total charge there is to distribute */
  real tot_ch_salt = con_salt / 2.0 * fluid_sites;
  real tot_ch_fl_p, tot_ch_fl_m;
  /* We assumed no walls, so total fluid charge is only due to salt and objs */
  if( objects_charge >= 0.0 ){ 
    tot_ch_fl_p = tot_ch_salt;
    tot_ch_fl_m = tot_ch_salt + objects_charge;
  }else{
    tot_ch_fl_p = tot_ch_salt - objects_charge;
    tot_ch_fl_m = tot_ch_salt;
  }
  /* Work out charge to put on each site normalized with probability proportional to exp value of potential phi */
  real factor = params.ch_bjl * PIG;
  real check_p = 0.0;
  real check_m = 0.0;
  real placed_f_charge = 0.0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        if( site_type[index][k] == FLUID ){
          /* Define normalized probability */
          real prob_here_m = exp( + dh[index][k] ) / norm_minus; 
          real prob_here_p = exp( - dh[index][k] ) / norm_plus;
          check_p += prob_here_p;
          check_m += prob_here_m;
  	      c_plus[index][k]  = tot_ch_fl_p * prob_here_p;
  	      c_minus[index][k] = tot_ch_fl_m * prob_here_m;
	        placed_f_charge += c_plus[index][k] - c_minus[index][k];
        }
       anormf0 += factor * ( c_plus[index][k] + c_minus[index][k] );
      }
    }
  }
      printf( "Placed EXPONENTIALLY a total of + %f and - %f salt+cions charge\nResulting NET FLUID charge is %f\n", tot_ch_fl_p, tot_ch_fl_m, placed_f_charge );
      printf("Integral of charge probabilities + %f - %f \n", check_p, check_m );
}

void check_electroneutrality( double ** c_plus, double ** c_minus, int ** site_type )
{
  int i, j, k;
  int siz_y = params.Ny + 2;

  /* Check electroneutrality */
  real tot_ch_p = 0.0;
  real tot_ch_n = 0.0;
  real tot_ch_p_f = 0.0;
  real tot_ch_n_f = 0.0;
  real tot_ch_p_o = 0.0;
  real tot_ch_n_o = 0.0;
  real tot_ch_p_w = 0.0;
  real tot_ch_n_w = 0.0;
  real tot_ch_p_c = 0.0;
  real tot_ch_n_c = 0.0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        tot_ch_p += c_plus[index][k];
        tot_ch_n += c_minus[index][k];
        if( site_type[index][k] == FLUID ){
          tot_ch_p_f += c_plus[index][k];
          tot_ch_n_f += c_minus[index][k];
        }
        if( site_type[index][k] == WALL ){
          tot_ch_p_w += c_plus[index][k];
          tot_ch_n_w += c_minus[index][k];
        }
        if( site_type[index][k] == SOLID ){
          tot_ch_p_o += c_plus[index][k];
          tot_ch_n_o += c_minus[index][k];
        }
        if( site_type[index][k] == CORNER ){
          tot_ch_p_c += c_plus[index][k];
          tot_ch_n_c += c_minus[index][k];
        }
      }
    }
  }
  real tot_ch = tot_ch_p - tot_ch_n;

  if( fabs( tot_ch ) > 10e-6 ) { 
    printf( "\t\tELECTRONEUTRALITY SCREWED. GOT %.10f TOTAL CHARGE (+%e -%e). CHECK WHAT IS WRONG! BYE\n\n", 
        tot_ch, tot_ch_p, tot_ch_n ); 
        //exit ( 0 ); 
  }else{
    printf( "\t\tELECTRONEUTRALITY OK. GOT TOTAL CHARGE OF %e\n", tot_ch );
    printf( "\t\tPUT %e CHARGE on FLUID (+%e %e)\n",  tot_ch_p_f - tot_ch_n_f, tot_ch_p_f, tot_ch_n_f );
    printf( "\t\tPUT %e CHARGE on SOLID (+%e %e)\n",  tot_ch_p_o - tot_ch_n_o, tot_ch_p_o, tot_ch_n_o );
    printf( "\t\tPUT %e CHARGE on WALL (+%e %e)\n",   tot_ch_p_w - tot_ch_n_w, tot_ch_p_w, tot_ch_n_w );
    printf( "\t\tPUT %e CHARGE on CORNERS (+%e %e)\n",   tot_ch_p_c - tot_ch_n_c, tot_ch_p_c, tot_ch_n_c );
  }
  printf("\n");
}

extern int x_dir[18], y_dir[18], z_dir[18];
real check_zeta_midpoint( real ** phi, real ** c_plus, real ** c_minus, int ** site_type ){
  int siz_y = params.Ny + 2;

  sor2( params.Nx, params.Ny, params.Nz, c_minus, c_plus, phi );
  int i, j, k, l;
  int s = 0;
  real zeta_mid_tot = 0.0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        if( site_type[index][k] == SOLID ){
          /* Here do not check PBC as unlikely there is a SOLID-FLUID pair at boundary */
	        for( l = 0; l < 18; l++ ){  
	          int ip = i + x_dir[l];
	          int jp = j + y_dir[l];
	          int kp = k + y_dir[l];
            int indexp = ip * siz_y + jp;
            if( site_type[indexp][kp] == FLUID ){
              zeta_mid_tot +=  ( phi[index][k] + phi[indexp][kp] ) / 2.0;
              //printf("%d %d %d and nn %d %d %d with phis %f %f\n", i, j, k, ip, jp, kp, phi[index][k], phi[indexp][kp] );
              s++;
            }
            //printf("%f %d\n", zeta_mid_tot, s);
          }
        }
      }
    }
  }
  return zeta_mid_tot / s;
}

#define x_shift 16
#define y_shift 8
#define z_shift 0

static inline void do_dual( int i, int j, int k, int * dual )
{
  i = ( i > params.Nx )? 1: i;
  i = ( i < 1 )? params.Nx: i;
  j = ( j > params.Ny )? 1: j;
  j = ( j < 1 )? params.Ny: j;
  k = ( k > params.Nz - 1 )? 0: k;
  k = ( k < 0 )? params.Nz - 1: k;
  *dual = i << x_shift | j << y_shift | k << z_shift;  
}

static inline void reverse_dual( int dual, int * i, int * j, int * k )
{
  int x_mask, y_mask, z_mask;
  x_mask = y_mask = z_mask = (1 << y_shift) - 1;

  *i =  dual >> x_shift & x_mask; 
  *j =  dual >> y_shift & y_mask; 
  *k =  dual >> z_shift & z_mask; 
}

int mc_sites;
int create_dual( int * dual )
{
  int i, j, k;

  int s = 0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      for( k = 0; k < params.Nz; k++ ){
        do_dual( i, j, k, &dual[s] );
        /* Checks */
        //int a, b ,c;
        //reverse_dual( dual[s], &a, &b, &c );
        //printf( "Mapping <-> Reverse mapping %d %d %d > %d (dual %d) > %d %d %d\n", i, j, k, s, dual[s], a, b, c );
        s++;
      }
    }
  }
  mc_sites = s;
  printf( "Number of MC sites is %d\n", mc_sites );
  return s;
}

void  create_distances( real cubic_cutoff, int * dual, real ** distance )
{
  int a, b;
  for( a = 0; a < mc_sites; a++ ){
  //for( a = 0; a < 100; a++ ){
    int d_a = dual[a];
    int i, j, k;
    reverse_dual( d_a, &i, &j, &k );
    //printf("Site %d with dual %d\n", a, d_a );
    for( b = 0; b < mc_sites; b++ ){
    //for( b = 0; b < 100; b++ ){
      int d_b = dual[b];
      int m, n, o;
      reverse_dual( d_b, &m, &n, &o );
      real r_ab = pow( pow((m - i), 2.0) + pow((n - j), 2.0) + pow((o - k),2.0), 0.5 );
      //printf( "Site %d %d %d is %e far away from %d %d %d\n", i, j, k, r_ab, m, n, o );
      real inv_r_ab = 1 / r_ab;
      /* I put a very small number on the diagonal, as will set q to zero when calculating.
       * If something is wrong, this gives a big contribution, should be able to spot it
       */
      distance[a][b] = ( a != b )? inv_r_ab : 0.000000001;
    }
  }
}

static inline void do_nn( int a, int dual, int * nns )
{
  nns[0] = a; /* Self */
  int i, j, k;
  reverse_dual( dual, &i, &j, &k );
  /* PBC are taken care of directly in do_dual */
  do_dual( i + 1, j, k, nns + 1 );  
  do_dual( i - 1, j, k, nns + 2 );  
  do_dual( i, j + 1, k, nns + 3 );  
  do_dual( i, j - 1, k, nns + 4 );  
  do_dual( i, j, k + 1, nns + 5 );  
  do_dual( i, j, k - 1, nns + 6 );  
}

void create_nns( int mc_sites, int * dual, int ** nns )
{
  int a;
  for( a = 0; a < mc_sites; a++ ){
    do_nn( a, dual[a], nns[a] );
    int i, j, k;
    reverse_dual( dual[a], &i, &j, &k );
    //printf( "Site %d %d %d has as neighbours:\n", i, j, k );
    int b;
    for( b = 1; b <= 6; b++ ){
      int n_i, n_j, n_k;
      reverse_dual( nns[a][b], &n_i, &n_j, &n_k );
      //printf( "\t\tNN number %d is %d %d %d\n", b, n_i, n_j, n_k );
    }
  }
}

inline static real do_tri_distance( int i, int j, int k, int as, int bs, int cs, int an, int bn, int cn )
{
  /* These distances should be tabulated but if lattice is bigger than 40**3 too much memory  */
  real a1 = 1.0 / pow( pow((i - as), 2.0) + pow((j - bs), 2.0) + pow((k - cs),2.0), 0.5 );
  real a2 = 1.0 / pow( pow((i - an), 2.0) + pow((j - bn), 2.0) + pow((k - cn),2.0), 0.5 );
  return a1 - a2;
}

real calc_delta_energy( double ** c_plus, double ** c_minus, int as, int bs, int cs, int an, int bn, int cn, real delta_ch, real try_delta_ch )
{
  int siz_y = params.Ny + 2;

  int i, j, k;
  int dual_s, dual_n;
  do_dual( as, bs, cs, &dual_s );
  do_dual( an, bn, cn, &dual_n );
  real dE = 0.0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        int dual_here;
        do_dual( i, j, k, &dual_here );
        /* Check this site is not one in the pair */
        if( dual_here != dual_s && dual_here != dual_n ){
          real tri_dist_here = do_tri_distance( i, j, k, as, bs, cs, an, bn, cn ); 
          real q_here = c_plus[index][k] - c_minus[index][k];
          dE += q_here * tri_dist_here;
        }
      }
    }
  }
  /* Add last term */
  dE -= ( delta_ch + try_delta_ch ); 
  /* Multiply per constant */
  dE *= params.ch_bjl * params.kbt * try_delta_ch;
  return dE;
}

void print_profile( real ** c_plus, real ** c_minus )
{
  int siz_y = params.Ny + 2;
  int index  = (params.Nx / 2) * siz_y + params.Ny / 2;
  int k;
  for( k = 0; k < params.Nz; k++ ){
    printf("%d %e\n", k, (c_plus[index][k] - c_minus[index][k]) );
  }
}

void mc_moves( real ** c_plus, real ** c_minus, int ** nns, int ** site_type, int * dual )
{
  int siz_y = params.Ny + 2;

  int m;
  real totE = 0.0;
  int accepted = 0;
  int moves    = 0;
  print_profile( c_plus, c_minus );
  for( m = 0; m < params.numsteps; m++){

    /* Choose random pair */
    int s  =  (int)( mc_sites * ran1((long *) &idum) );
    int n  =  (int)( 6 * ran1((long *) &idum) ) + 1;
    int as, bs, cs;
    reverse_dual( dual[s], &as, &bs, &cs );
    int index  = as * siz_y + bs;
    //printf( "Selected for move %d (%d %d %d) and ", s, as, bs, cs );
    int an, bn, cn;
    int nn =  nns[s][n];
    reverse_dual( nn, &an, &bn, &cn );
    int index_nn = an * siz_y + bn;
    //printf( "%d-%d (%d %d %d) ", n, nn, an, bn, cn );
    /* Check it is a pair of FLUID */
    if( !( site_type[index][cs] == FLUID && site_type[index_nn][cn] == FLUID) ){ 
      continue;
    }

    real ch_s  = c_plus[index][cs]  - c_minus[index][cs];
    real ch_nn = c_plus[index_nn][cn] - c_minus[index_nn][cn];
    /* delta ch can be up to more or less the charge difference between pair sites */
    real delta_ch = ( ch_s - ch_nn );
    real try_delta_ch = 2.0 * delta_ch * ( ran1((long *) &idum) - 0.5 );
    //printf( "with ch %e and %e delta_ch %e try_delta_ch %e\n", ch_s, ch_nn, delta_ch, try_delta_ch );
    /* Security check not to allow an unphysical trial move, i.e. placing in nn more than the charge in s or viceversa */
    if( fabs(try_delta_ch) >= fabs(ch_s) || fabs(try_delta_ch) >= fabs(ch_nn) ){ 
      //printf("Continuing %d %d %d and %d %d %d as we got with chs %e and ch_nn %e delta_ch %e and try_delta_ch %e\n", as, bs, cs, an, bn, cn, ch_s, ch_nn, delta_ch, try_delta_ch ); 
        continue; }

    moves++;
    /* Calculate delta elec energy of trial move */
    real deltaE = calc_delta_energy( c_plus, c_minus, as, bs, cs, an, bn, cn, delta_ch, try_delta_ch );
    printf("deltaE trial is %e\n", deltaE);

    /* Accept ? update : continue; */
    real ran = ran1((long *) &idum);
    if( deltaE < 0.0 || ran < exp( -deltaE / params.kbt ) ){ 
      real add = try_delta_ch / 2.0;
      c_plus[index][cs]  = c_plus[index][cs]  + add;
      c_minus[index][cs] = c_minus[index][cs] - add;
      c_plus[index_nn][cn]  = c_plus[index_nn][cn]  - add;
      c_minus[index_nn][cn] = c_minus[index_nn][cn] + add;
      totE += deltaE;
      accepted++;
      //printf("Move %d with delta_E %e accepted (tot %d). totE %e\n", m, deltaE, accepted, totE );
    }else{
      //printf("REJECTED\n");
    }
    if( m % params.outfreq == 0 ){ 
      /* Did not know that float ( int / int ) produces SIGFPE, so double casting */
      printf( "\nTot_energy move %d is %e. Net moves %d. Acceptance rate %f\n", m, totE, moves, (float) accepted / (float) moves  );
      //print_profile( c_plus, c_minus );
    }    
  }
}

void mc_equilibration( double ** c_plus, double ** c_minus, real ** dh, object  * objects, int ** site_type )
{
  int go = 1;
  real init_zeta = params.r_pair;
  real ** phi   = mem_r_lattice2( (params.Nx + 2) * (params.Ny + 2), params.Nz + 2 ); 
  while( go ){
    dh_potential( dh, init_zeta, params.ch_lambda, objects ) ;
    printf( "\n\t\tMC-Placing charges exponentially with init_zeta %f\n", init_zeta );
    place_charges_exp( c_plus, c_minus, dh, site_type );
    real eff_zeta = check_zeta_midpoint( phi, c_plus, c_minus, site_type );
    double tol_zeta = ( eff_zeta - init_zeta ) / eff_zeta;
    printf( "\t\tMC-Eff-zeta is %f, tolerance %f\n", eff_zeta, tol_zeta );
    if( fabs(tol_zeta) < 0.05 ){ 
      printf( "\t\tMC-Tolerance  less than one 1%%.Done\n" ); go = 0; 
    }else{
      real reinit_zeta = init_zeta + ( eff_zeta - init_zeta ) / 10.0;
      /* It turns out that for high potential, starting from low potential, makes
       * eff_zeta very big, so that restarting from one big make next eff_zeta lower
       * than the input init_zeta or negative! So, protocol should start from low init_zeta 1.0
       * then going up smoothly (now put / 10.0 but can change) as here */
      printf( "\t\tMC-Restarting with init-zeta %f\n", reinit_zeta );
      init_zeta = reinit_zeta;
    }
  }
  free_2D( (void **) phi );

  int  * dual      = (int *)  calloc( params.Nx * params.Ny * params.Nz, sizeof(int) );
  create_dual( dual );
  int ** nns = (int **) mem_i_lattice2( mc_sites, 6 );
  create_nns( mc_sites, dual, nns );
  mc_moves( c_plus, c_minus, nns, site_type, dual );
}
