#include <stdlib.h>
#include <math.h>
#include "simpar.h"
#include "lb.h"
#include "init.h" // This is for gasdev

extern sim_params_t params;
extern int fluid_nodes;
extern long int idum;

float a0[18]  = { a_01, a_01, a_01, a_01, a_01, a_01,
                       a_02, a_02, a_02, a_02, a_02, a_02,
                       a_02, a_02, a_02, a_02, a_02, a_02 };
float a1[18]  = { a_11, a_11, a_11, a_11, a_11, a_11,
                       a_12, a_12, a_12, a_12, a_12, a_12,
                       a_12, a_12, a_12, a_12, a_12, a_12 };
/* _plus means that the corresponding component of the lattice velocity is positive, 
and "_minus" that it is negative */
int  x_plus[5]  = { 0, 6, 8, 10, 12 };
int  x_minus[5] = { 1, 7, 9, 11, 13 };
int  y_plus[5]  = { 2, 6, 9, 14, 16 };
int  y_minus[5] = { 3, 7, 8, 15, 17 };
int  z_plus[5]  = { 4, 10, 13, 14, 17 };
int  z_minus[5] = { 5, 11, 12, 15, 16 };
/* useful for boundary conditions. It refers to the velocity 
that has opposite direction. For example, the velocity 1 is the 
one opposite to velocity 0; or 14 and 15 correspond to opposite 
velocities */
int l_opp[18] = { 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16 };
/* Definition of the geometry: each node is connected to 18 neighbours. 
x_dir, y_dir and z_dir correspond to the x,y and z components 
of such vectors */
int x_dir[18] = {1, -1, 0, 0, 0, 0,
                           1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0};
int y_dir[18] = {0, 0, 1, -1, 0, 0,
                           1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1};
int z_dir[18] = {0, 0, 0, 0, 1, -1,
                           0,  0,  0,  0, 1, -1, -1,  1, 1, -1, -1,  1};
/* weights of the velocities (i.e. the previous 18 vectors 
connecting neighbors. These weights appear in the equilibrium 
distribution and ensure the conservation of mass and momentum 
and the proper symmetries to get hydrodynamics */
int weighting[18] = {2, 2, 2, 2, 2, 2,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
/* list used in the trtment of boundary nodes: they are stored using binary operations*/
int x_mask  = (1 << x_num_bit ) - 1;
int y_mask  = (1 << y_num_bit ) - 1;
int z_mask  = (1 << z_num_bit ) - 1;
int x_shift = 7 + z_num_bit + y_num_bit;
int y_shift = 7 + z_num_bit;
int z_shift = 7;

vector ff_ext;
double delta[18]; // init in a function to get precise sqrt(2)

void init_delta( double * delta )
{
  int i;
  delta[0]=0.0;
  for( i = 0; i <= 5; i++ ) delta[i] = 1.0;
  for( i = 6; i < 18; i++ ) delta[i] = sqrt( 2.0 );
}

/*-------------------------------------------------------------------------*/
void  lbe_initvel( real *** velcs_ptr, vector uu, int *site_type[])
/*-------------------------------------------------------------------------*/
{
  int      x, y, z, velc_dir, link_dir, index;  
  real    vx, vy,vz, vx1, vy1, vz1;
  real max_x    = params.Nx;
 
  int siz_y = params.Ny + 2;
  /* set initial velocity to study vacf: the fluid is at rest with uniform density */
  vx = uu.x * 0.7;
  vx = 0.0;
  vy = uu.y * 0.7;
  vy = 0.0;
  vz = uu.z * 0.7;
  vz = 0.0;
  /* here we set the density to 18 (it will become 24 when we take into account that velocities moving along the coordinate axis have an extra factor of 2 (see the last loop in this subroutine)*/
  for(x = 1; x <= max_x; x++){
    for(y = 1; y <= params.Ny; y++){
      index = x*siz_y + y;
      for(velc_dir = 0; velc_dir < 18; velc_dir++){
	for(z = 0; z < params.Nz; z++ ){ 
	  *(velcs_ptr[index][velc_dir]+z) = 1.0;                  
	}
      }
      for(velc_dir = 0; velc_dir < 5; velc_dir++){
	/* initial velocity profile in the y direction: velocity applied only at one node */
	for(z = 0; z < params.Nz; z++ ){ 
	  vx1 = vx*gasdev( (long *) &idum );
	  vy1 = vy*gasdev( (long *) &idum );
	  vz1 = vz*gasdev( (long *) &idum );
	  if(site_type[index][z]==FLUID){
	    link_dir = x_plus[velc_dir];
	    *(velcs_ptr[index][link_dir]+z) += vx1/10.0;   
	    link_dir = x_minus[velc_dir];
	    *(velcs_ptr[index][link_dir]+z) -= vx1/10.0;  
	    link_dir = y_plus[velc_dir];
	    *(velcs_ptr[index][link_dir]+z) += vy1/10.0;   
	    link_dir = y_minus[velc_dir];
	    *(velcs_ptr[index][link_dir]+z) -= vy1/10.0;  
	    link_dir = z_plus[velc_dir];
	    *(velcs_ptr[index][link_dir]+z) += vz1/10.0;   
	    link_dir = z_minus[velc_dir];
	    *(velcs_ptr[index][link_dir]+z) -= vz1/10.0;  
	  } 
	}    
      }
    }  
  }
  for(x = 1; x <= max_x; x++){
    for(y = 1; y <= params.Ny; y++){
      index = x*siz_y + y;
      for(velc_dir = 0; velc_dir < 6; velc_dir++){
	for(z = 0; z < params.Nz; z++ ){ 
	  *(velcs_ptr[index][velc_dir]+z) = 
	    *(velcs_ptr[index][velc_dir]+z)*2.0;
	}
      }
    }
  }
}
/*---------------------------------------------------------------------*/
void  lbe_bconds( real *** velcs_ptr )
/*---------------------------------------------------------------------*/
{
  int   x, y, xy_new, xy_old, link_dir, velc_dir;
  int siz_x = params.Nx + 2;
  int siz_y = params.Ny + 2;

  for(velc_dir = 0; velc_dir < 5; velc_dir ++){
    link_dir = x_plus[velc_dir];
    for( y = 1; y <= params.Ny; y++){
      xy_new = y;
      xy_old = params.Nx*siz_y + y;
      velcs_ptr[xy_new][link_dir] = velcs_ptr[xy_old][link_dir];
    }
    link_dir = x_minus[velc_dir];
    for( y = 1; y <= params.Ny; y++){
      xy_new = (params.Nx +1)*siz_y + y;
      xy_old = siz_y + y;
      velcs_ptr[xy_new][link_dir] = velcs_ptr[xy_old][link_dir];
    }
    link_dir = y_plus[velc_dir];
    for( x = 0; x < siz_x; x++){
      xy_new = x*siz_y;
      xy_old = x*siz_y + params.Ny;
      velcs_ptr[xy_new][link_dir] = velcs_ptr[xy_old][link_dir];
    }
    link_dir = y_minus[velc_dir];
    for( x = 0; x < siz_x; x++){
      xy_new = x*siz_y+params.Ny+1;
      xy_old = x*siz_y + 1;
      velcs_ptr[xy_new][link_dir] = velcs_ptr[xy_old][link_dir];
    }
  }
}

/*---------------------------------------------------------------------*/
void  lbe_movexy( real *** velcs_ptr )
/*---------------------------------------------------------------------*/
{
  int x, y, xy_new, xy_old, link_dir, velc_dir;
  int siz_y = params.Ny + 2;
  
  /* each distribution function moves from the node it is to the corrresponding neighbor depending on its velocity */
  for(velc_dir = 0; velc_dir < 5; velc_dir++){
    link_dir = x_plus[velc_dir];
    for(x = params.Nx; x >= 1; x--){
      for(y = 0; y < siz_y; y++){
    	  xy_new = x*siz_y + y;
	      xy_old = xy_new - siz_y;
	      velcs_ptr[xy_new][link_dir] = velcs_ptr[xy_old][link_dir];
      }
    }
    link_dir = x_minus[velc_dir];
    for(x = 1; x <= params.Nx ; x++){
      for(y = 0; y < siz_y; y++){
	      xy_new = x*siz_y + y;
	      xy_old = xy_new + siz_y;
	      velcs_ptr[xy_new][link_dir] = velcs_ptr[xy_old][link_dir];
      }
    }
    
    link_dir = y_plus[velc_dir];
    for(x = 1; x <= params.Nx; x++){
      for(y = params.Ny; y >= 1; y--){
	      xy_new = x*siz_y + y;
	      xy_old = xy_new - 1;
	      velcs_ptr[xy_new][link_dir] = velcs_ptr[xy_old][link_dir];
      }
    }
    link_dir = y_minus[velc_dir];
    for(x = 1; x <= params.Nx; x++){
      for(y = 1; y <= params.Ny; y++){
	      xy_new = x*siz_y + y;
	      xy_old = xy_new + 1;
	      velcs_ptr[xy_new][link_dir] = velcs_ptr[xy_old][link_dir];
      }
    }
  }
}

/*--------------------------------------------------------------------*/
real  do_z_column( real ** velcz_ptr, real nu, int x, int y, 
       vector * ptot, vector  *str, real   *solute_force[3], real *jx, 
       real   *jy, real   *jz, real   *rho, int *site_type )
/*---------------------------------------------------------------------*/	
{
/* this column is called do_z because it relaxes the distribution along all values of 
z for a given x and y values. This is done in this way just because of numerical convenience. 
It is a good way to distribute the memory load in the simulation*/

  static int     num_mode;
  static real   nu_factor, *modes_ptr;
  real   m, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, rho_inverse;
  real   m09_a, m09_b, m45_a, m45_b, m42, m52, p0_tot;
  int     z, z_plus, z_minus, z_ptr;

  num_mode = 10;
  modes_ptr = (real *) malloc( params.Nz * num_mode * sizeof(real) );
  nu_factor = nu;
  /*
  if(initialize_chk == 0){
    num_mode = 10;
    nu_factor = nu;
    modes_ptr = (real *) malloc( params.Nz * num_mode * sizeof(real) );
    initialize_chk = 1;
  }
  */

  p0_tot = 0.0;
  (*ptot).x = (*ptot).y=(*ptot).z =0.0;
  /*collision is executed expressing the relaxed densities in terms of density and  momentum (cf. J.Fluid Mech. (1994-1995) by Ladd, or the review paper with Verberg in J. Stat. Phys.). Hence, I first compute the relevant moments.
Three types of fluid: FLUID_TYPE=0 assumes viscosity 1/6. FLUID_TYPE=1 does not incorporate the convection term in LB; it can be used for Re=0; FLUID_TYPE=2 does not make any approximation. Probably the one preferred for this code */
  for(z = 0, z_ptr = 0; z < params.Nz; z++, z_ptr += num_mode){
    z_plus  = mod( z+1, params.Nz );
    z_minus = mod( z-1+ params.Nz, params.Nz );
    

#       if FLUID_TYPE == 0

    m = *(velcz_ptr[0]+z);        m0  = m; m1  = m;
    m = *(velcz_ptr[1]+z);        m0 += m; m1 -= m;
    m = *(velcz_ptr[2]+z);        m0 += m; m2  = m;
    m = *(velcz_ptr[3]+z);        m0 += m; m2 -= m;
    m = *(velcz_ptr[4]+z_minus);  m0 += m; m3  = m;
    m = *(velcz_ptr[5]+z_plus );  m0 += m; m3 -= m;
    m = *(velcz_ptr[6]+z);        m0 += m; m1 += m; m2 += m;
    m = *(velcz_ptr[7]+z);        m0 += m; m1 -= m; m2 -= m;
    m = *(velcz_ptr[8]+z);        m0 += m; m1 += m; m2 -= m;
    m = *(velcz_ptr[9]+z);        m0 += m; m1 -= m; m2 += m;
    m = *(velcz_ptr[10]+z_minus); m0 += m; m1 += m; m3 += m;
    m = *(velcz_ptr[11]+z_plus ); m0 += m; m1 -= m; m3 -= m;  
    m = *(velcz_ptr[12]+z_plus ); m0 += m; m1 += m; m3 -= m;
    m = *(velcz_ptr[13]+z_minus); m0 += m; m1 -= m; m3 += m;
    m = *(velcz_ptr[14]+z_minus); m0 += m; m2 += m; m3 += m;
    m = *(velcz_ptr[15]+z_plus ); m0 += m; m2 -= m; m3 -= m;
    m = *(velcz_ptr[16]+z_plus ); m0 += m; m2 += m; m3 -= m;
    m = *(velcz_ptr[17]+z_minus); m0 += m; m2 -= m; m3 += m;
#       else
    m  = *(velcz_ptr[0]+z);
    m0 = m; m1 = m; m4 = m; m5 = m;
    m  = *(velcz_ptr[1]+z);
    m0 += m; m1 -= m; m4 += m; m5 += m;
    m  = *(velcz_ptr[2]+z);
    m0 += m; m2 = m; m5 -= m;
    m  = *(velcz_ptr[3]+z);
    m0 += m; m2 -= m; m5 -= m;
    m  = *(velcz_ptr[4]+z_minus);
    m0 += m; m3 = m; m4 -= m;
    m  = *(velcz_ptr[5]+z_plus);
    m0 += m; m3 -= m; m4 -= m;
    m  = *(velcz_ptr[6]+z);
    m0 += m; m1 += m; m2 += m; m4 += m; m6  = m;
    m  = *(velcz_ptr[7]+z);
    m0 += m; m1 -= m; m2 -= m; m4 += m; m6 += m;
    m  = *(velcz_ptr[8]+z);
    m0 += m; m1 += m; m2 -= m; m4 += m; m6 -= m;
    m  = *(velcz_ptr[9]+z);
    m0 += m; m1 -= m; m2 += m; m4 += m; m6 -= m;
    m  = *(velcz_ptr[10]+z_minus);
    m0 += m; m1 += m; m3 += m; m5 += m; m7  = m;
    m  = *(velcz_ptr[11]+z_plus);
    m0 += m; m1 -= m; m3 -= m; m5 += m; m7 += m;
    m  = *(velcz_ptr[12]+z_plus);
    m0 += m; m1 += m; m3 -= m; m5 += m; m7 -= m;
    m  = *(velcz_ptr[13]+z_minus);
    m0 += m; m1 -= m; m3 += m; m5 += m; m7 -= m;
    m  = *(velcz_ptr[14]+z_minus);
    m0 += m; m2 += m; m3 += m; m4 -= m; m5 -= m; m8  = m;
    m  = *(velcz_ptr[15]+z_plus);
    m0 += m; m2 -= m; m3 -= m; m4 -= m; m5 -= m; m8 += m;
    m  = *(velcz_ptr[16]+z_plus);
    m0 += m; m2 += m; m3 -= m; m4 -= m; m5 -= m; m8 -= m;
    m  = *(velcz_ptr[17]+z_minus);
    m0 += m; m2 -= m; m3 += m; m4 -= m; m5 -= m; m8 -= m;
    m4 += m5; m5 = m4 - m5*2;
    m4 *= nu; m5 *= nu; m6 *= nu;
    m7 *= nu; m8 *= nu; m9  = 0.0;
#       if FLUID_TYPE == 2
    rho_inverse = 1.0/(m0);
        
    m4 += (2.0*m1*m1 - m2*m2 - m3*m3)*rho_inverse*(1.0 - nu_factor); 
    m5 += (            m2*m2 - m3*m3)*rho_inverse*(1.0 - nu_factor);
    m6 += (m1*m2                    )*rho_inverse*(1.0 - nu_factor);
    m7 += (m1*m3                    )*rho_inverse*(1.0 - nu_factor);
    m8 += (m2*m3                    )*rho_inverse*(1.0 - nu_factor);
    m9 += (    m1*m1 + m2*m2 + m3*m3)*rho_inverse*2.0;
#       endif 

#       endif
    /* mo=density, m1=x-component of momentum, m2 = y-component, m3 z-component */
    *( modes_ptr + z_ptr ) = m0;
    if(*(site_type+z)==FLUID){
      *(modes_ptr + z_ptr + 1) = m1 + *(solute_force[0]+z) + ff_ext.x;
      *(modes_ptr + z_ptr + 2) = m2 + *(solute_force[1]+z) + ff_ext.y;
      *(modes_ptr + z_ptr + 3) = m3 + *(solute_force[2]+z) + ff_ext.z;
      //if( (x == 1 || x == 2) && y == 5 && z == 5 ) printf("SF_x %d %d %d is %e and ff_gravity %e\n", x, y, z, *(solute_force[0]+z), ff_ext.x );
      //printf("FLUID pos %d solute force %e %e %e gravity force %e %e %e\n",z, *(solute_force[0]+z), *(solute_force[1]+z), *(solute_force[2]+z), ff_ext.x, ff_ext.y, ff_ext.z );
    }else{
      *(modes_ptr + z_ptr + 1) = m1 + *(solute_force[0]+z);
      *(modes_ptr + z_ptr + 2) = m2 + *(solute_force[1]+z);
      *(modes_ptr + z_ptr + 3) = m3 + *(solute_force[2]+z);
      //printf("SOLID pos %d solute force %e %e %e\n",z, *(solute_force[0]+z), *(solute_force[1]+z), *(solute_force[2]+z));
    }
    /* here ptot is the kinetic energy, but you can moodify it to get any other vectrial quantity of interest */
    p0_tot +=    m2;
    (*ptot).x += m1*m1;
    (*ptot).y += m2*m2;
    (*ptot).z += m3*m3;
    *(rho+z)= m0;
    *(jx+z) = m1;
    *(jy+z) = m2;
    *(jz+z) = m3;
#       if FLUID_TYPE != 0
    /* the rest of moments related to different components of stress tensor */
    *(modes_ptr + z_ptr + 4) = m4;
    *(modes_ptr + z_ptr + 5) = m5;
    *(modes_ptr + z_ptr + 6) = m6;
    *(modes_ptr + z_ptr + 7) = m7;
    *(modes_ptr + z_ptr + 8) = m8;
    *(modes_ptr + z_ptr + 9) = m9;
#       endif
  }  /* End loop over z column */
  /*once the moments have been computed, I will update the distribution function */
  for( z = 0, z_ptr = 0; z < params.Nz; z++, z_ptr += num_mode ){
    /*here the moments are scaled by the approrpiate prefactors coming from the equilibrium distribution function (cf. Ladd again)*/
    m0 = *(modes_ptr + z_ptr    )/24.0;
    m1 = *(modes_ptr + z_ptr + 1)/12.0;
    m2 = *(modes_ptr + z_ptr + 2)/12.0;
    m3 = *(modes_ptr + z_ptr + 3)/12.0;
#       if FLUID_TYPE != 0

    m4 = *(modes_ptr + z_ptr + 4)/48.0;
    m5 = *(modes_ptr + z_ptr + 5)/16.0;
    m6 = *(modes_ptr + z_ptr + 6)/ 4.0;
    m7 = *(modes_ptr + z_ptr + 7)/ 4.0;
    m8 = *(modes_ptr + z_ptr + 8)/ 4.0;
    m9 = *(modes_ptr + z_ptr + 9)/24.0;
#       endif


#       if FLUID_TYPE == 0
	
    *(velcz_ptr[0]+z) = (m0 + m1)*2.0;
    *(velcz_ptr[1]+z) = (m0 - m1)*2.0;
    *(velcz_ptr[2]+z) = (m0 + m2)*2.0;
    *(velcz_ptr[3]+z) = (m0 - m2)*2.0;
    *(velcz_ptr[4]+z) = (m0 + m3)*2.0;
    *(velcz_ptr[5]+z) = (m0 - m3)*2.0;
    *(velcz_ptr[6]+z) =  m0 + m1 + m2;
    *(velcz_ptr[7]+z) =  m0 - m1 - m2;
    *(velcz_ptr[8]+z) =  m0 + m1 - m2;
    *(velcz_ptr[9]+z) =  m0 - m1 + m2;
    *(velcz_ptr[10]+z)=  m0 + m1 + m3;
    *(velcz_ptr[11]+z)=  m0 - m1 - m3;
    *(velcz_ptr[12]+z)=  m0 + m1 - m3;
    *(velcz_ptr[13]+z)=  m0 - m1 + m3;
    *(velcz_ptr[14]+z)=  m0 + m2 + m3;
    *(velcz_ptr[15]+z)=  m0 - m2 - m3;
    *(velcz_ptr[16]+z)=  m0 + m2 - m3;
    *(velcz_ptr[17]+z)=  m0 - m2 + m3;
    
#       else
    m09_a = (m0 - m9)*2.0; m09_b = m0 + m9;
    m45_a = m4 + m5; m45_b = m4 - m5;
    m42   = 2.0*m4; m52 = 2.0*m5;

     /* here new values of distribution function finally computed according to the relaxation rule to equilibrium */
    *(velcz_ptr[0]+z) = m09_a + m1*2.0 + m4*4.0;
    *(velcz_ptr[1]+z) = m09_a - m1*2.0 + m4*4.0;
    *(velcz_ptr[2]+z) = m09_a + m2*2.0 - m42    +m52;
    *(velcz_ptr[3]+z) = m09_a - m2*2.0 - m42    +m52;
    *(velcz_ptr[4]+z) = m09_a + m3*2.0 - m42    -m52;
    *(velcz_ptr[5]+z) = m09_a - m3*2.0 - m42    -m52;
    *(velcz_ptr[6]+z) = m09_b + m1     + m2     +m45_a + m6;
    *(velcz_ptr[7]+z) = m09_b - m1     - m2     +m45_a + m6;
    *(velcz_ptr[8]+z) = m09_b + m1     - m2     +m45_a - m6;
    *(velcz_ptr[9]+z) = m09_b - m1     + m2     +m45_a - m6;
    *(velcz_ptr[10]+z)= m09_b + m1     + m3     +m45_b + m7;
    *(velcz_ptr[11]+z)= m09_b - m1     - m3     +m45_b + m7;
    *(velcz_ptr[12]+z)= m09_b + m1     - m3     +m45_b - m7;
    *(velcz_ptr[13]+z)= m09_b - m1     + m3     +m45_b - m7;
    *(velcz_ptr[14]+z)= m09_b + m2     + m3     -m42   + m8;
    *(velcz_ptr[15]+z)= m09_b - m2     - m3     -m42   + m8;
    *(velcz_ptr[16]+z)= m09_b + m2     - m3     -m42   - m8;
    *(velcz_ptr[17]+z)= m09_b - m2     + m3     -m42   - m8;
#       endif
  } /* End loop over column */
  free( modes_ptr );
  return(p0_tot);   
}

/*---------------------------------------------------------------------*/
void  wallz_plus_bc( real *** velcs_ptr, int x_pos )
/*---------------------------------------------------------------------*/
{
  int y, x, lk, l1, l2, x1, y1, z1, x2, y2, z2, xy1, xy2;
  real f1, f2, uw, ampl;
  int siz_y = params.Ny + 2;
  
  /* you can have a moving wall if wall_vel != 0*/
  uw = params.wall_vel;
  z1 = x_pos;
  for ( lk = 0; lk < 5; lk++ ){
    l1 = z_minus[lk];
    l2 = l_opp[l1];
    z2 = z1 + z_dir[l1];
    z2 = mod( z2 + params.Nz, params.Nz );
    for ( x = 1; x <= params.Nx; x++ ){
      x1 = x;
      x2 = x1 + x_dir[l1];
      x2 = mod( x2 + params.Nx -1, params.Nx ) + 1;
      for ( y = 1; y <= params.Ny; y++ ){
	      y1 = y;
	      xy1 = x1 * siz_y + y1;
	      y2  = y1 + y_dir[l1];
	      y2 = mod( y2 + params.Ny - 1, params.Ny ) + 1;
	      xy2 = x2 * siz_y + y2;
	      ampl = 1.0;
	      if( l1 < 6 ) ampl = 2.0;
	        f1 = *( velcs_ptr[xy1][l1] + z1 ) + 2.0 * ( 24. / 12. ) * uw * y_dir[l1] * ampl;
	        f2 = *( velcs_ptr[xy2][l2] + z2 ) - 2.0 * ( 24. / 12. ) * uw * y_dir[l2] * ampl;
	      	*( velcs_ptr[xy1][l1] + z1 ) = f2;
	        *( velcs_ptr[xy2][l2] + z2 ) = f1;
      }
    }
  }
}
/*---------------------------------------------------------------------*/
void  wallz_minus_bc( real *** velcs_ptr, int x_pos )
/*---------------------------------------------------------------------*/
{
  int y, x, lk, l1, l2, x1, y1, z1, x2, y2, z2, xy1, xy2;
  real f1, f2, uw, ampl;
  int siz_y = params.Ny + 2;
  
  /* wall with velocity opposite to the previous one: Couette geometry */
  uw = -params.wall_vel;
  z1 = x_pos;

  for (lk = 0; lk < 5; lk++ ){
    l1=z_plus[lk];
    l2= l_opp[l1];
    z2= z1 + z_dir[l1];
    z2= mod(z2 + params.Nz, params.Nz);
    for (x = 1; x <= params.Nx; x++ ){
      x1 = x;
      x2 = x1 + x_dir[l1];
      x2 = mod(x2 + params.Nx -1, params.Nx) + 1;
      for (y = 1; y <= params.Ny; y++ ){
	y1 = y;
	xy1= x1*siz_y + y1;
	y2 = y1 + y_dir[l1];
	y2 = mod(y2 + params.Ny -1, params.Ny) + 1;
	xy2= x2*siz_y + y2;
     	ampl=1.0;
	if(l1<6) ampl=2.0;
	f1 = *(velcs_ptr[xy1][l1] + z1)+ 2.0*(24./12.)*uw*y_dir[l1]*ampl;
	f2 = *(velcs_ptr[xy2][l2] + z2)- 2.0*(24./12.)*uw*y_dir[l2]*ampl;
	
	*(velcs_ptr[xy1][l1] + z1) = f2;
	*(velcs_ptr[xy2][l2] + z2) = f1;
      }
    }
  }
}

/*---------------------------------------------------------------------*/
void  bnodes_f( real *** velcs_ptr, object * objects, object * objects_ptr, b_node * b_nodes_ptr, int n_obj ) 
/*---------------------------------------------------------------------*/
{
   
     real  fx1, fy1, fz1, f1, f2, ux, uy, uz, mass;
     real  fx2, fy2, fz2, rx, ry, rz, wx, wy , wz, u_node, inertia;
     real  ux_new, uy_new, uz_new, wx_new, wy_new,wz_new;
     real  fx3, fy3, fz3, c1, c2, c3;
     real  fx4, fy4, fz4, c4, c5, c6;
     int    x1, y1, z1, x2, y2, z2, l1, l2, xy1, xy2;
     int    ns, i_node, n_bnode, n;

     int siz_y = params.Ny + 2;

     real sum_1_x, sum_1_y, sum_1_z;
     real sum_2_x, sum_2_y, sum_2_z;
     real sum_3_x, sum_3_y, sum_3_z;
     real sum_4_x, sum_4_y, sum_4_z;


    ns = 0;         

    sum_1_x = sum_1_y = sum_1_z = 0.0;
    sum_2_x = sum_2_y = sum_2_z = 0.0;
    sum_3_x = sum_3_y = sum_3_z = 0.0;
    sum_4_x = sum_4_y = sum_4_z = 0.0;
    
    ux = objects_ptr->u.x;
    uy = objects_ptr->u.y;
    uz = objects_ptr->u.z;
    wx = objects_ptr->w.x;
    wy = objects_ptr->w.y;
    wz = objects_ptr->w.z;
    mass = objects_ptr->mass;
    inertia = objects_ptr->inertia;


    /* 
     * This is formula (14) JCP (103) Lowe Frenkel Masters, for an implicit integration/calculation  
     * of final velocity required to match final colloid and final fluid velocity, instead of a simpler
     * euler scheme
     */
    for( n_bnode = 0; n_bnode < b_nodes_ptr->num_bnode; n_bnode++ )
    {
       i_node = *( b_nodes_ptr->i_nodes + n_bnode );

       x1     = i_node >> x_shift & x_mask;
       y1     = i_node >> y_shift & y_mask;
       z1     = i_node >> z_shift & z_mask;
       l1     = i_node >> 2       & 31;
    
       x2     = x1 + x_dir[l1];
       y2     = y1 + y_dir[l1];
       z2     = z1 + z_dir[l1]; z2 = mod(z2 + params.Nz, params.Nz );
       l2     = l_opp[l1];
     
       xy1    = x1 * siz_y + y1;
       xy2    = x2 * siz_y + y2;  

       f1 = *(velcs_ptr[xy1][l1] + z1);
       f2 = *(velcs_ptr[xy2][l2] + z2);

       rx = *(b_nodes_ptr->r_x + n_bnode);
       ry = *(b_nodes_ptr->r_y + n_bnode);
       rz = *(b_nodes_ptr->r_z + n_bnode);
       n = weighting[l1];

       c1 = ry * z_dir[l1] - rz * y_dir[l1];
       c2 = rz * x_dir[l1] - rx * z_dir[l1];
       c3 = rx * y_dir[l1] - ry * x_dir[l1];

       c4 = c1 * c1;
       c5 = c2 * c2;
       c6 = c3 * c3;

       fx1     = 2.0 * ( f1 - f2 ) * x_dir[l1];
       fy1     = 2.0 * ( f1 - f2 ) * y_dir[l1];
       fz1     = 2.0 * ( f1 - f2 ) * z_dir[l1];
       fx2     = n * 8.0 * x_dir[l1] * x_dir[l1];
       fy2     = n * 8.0 * y_dir[l1] * y_dir[l1];
       fz2     = n * 8.0 * z_dir[l1] * z_dir[l1];

       fx3     = 2.0 * ( f1 - f2 ) * c1;
       fy3     = 2.0 * ( f1 - f2 ) * c2;
       fz3     = 2.0 * ( f1 - f2 ) * c3;
       fx4     = n * 8.0 * c4;
       fy4     = n * 8.0 * c5;
       fz4     = n * 8.0 * c6;

       sum_1_x += fx1;
       sum_2_x += fx2;
       sum_1_y += fy1;
       sum_2_y += fy2;
       sum_1_z += fz1;
       sum_2_z += fz2;
       
       sum_3_x += fx3;
       sum_4_x += fx4;
       sum_3_y += fy3;
       sum_4_y += fy4;
       sum_3_z += fz3;
       sum_4_z += fz4;
    }

    sum_1_x /= mass;
    sum_1_y /= mass;
    sum_1_z /= mass;
    sum_2_x /= mass;
    sum_2_y /= mass;
    sum_2_z /= mass;

    sum_3_x /= inertia;
    sum_4_y /= inertia;
    sum_3_z /= inertia;
    sum_4_x /= inertia;
    sum_3_y /= inertia;
    sum_4_z /= inertia;

    /* 
     * Calculation of new velocity after sum of forces over all link nodes. If there is an external force, 
     * This is added to the numerator
     */ 
    int fixed = 0;
    real push = params.u_x + params.u_y + params.u_z;
    /* In case of sedimentation or two particles, colloids are kept fixed */
    if( push > 0.0f || params.num_obj == 2 ) fixed = 1;
    /* In case colloids move, derive colloid-colloid interaction. Sorry, double counting! */
    vector f_cc;
    f_cc.x = f_cc.y = f_cc.z = 0.0;
    if( params.move && 0 ){
      /* Loop over other colloids, checking distance from this */
      int obj;
	    for( obj = 0; obj < params.num_obj; obj++ ){
        /* Skip self */
        if( objects_ptr->self == objects[obj].self ) continue;
        vector v_dist;
        real dist_this = rpbc_vec_dist( objects[obj].r.x, objects[obj].r.y, objects[obj].r.z, objects_ptr->r.x, objects_ptr->r.y, objects_ptr->r.z, &v_dist );
        if( dist_this < 0.5 ){
          printf("Colloids %d @ (%e %e %e) and %d (%e %e %e)are %e apart, less than cutoff 0.5 (hardcoded). Calculating CC force", objects_ptr->self - 10, objects[obj].r.x, objects[obj].r.y, objects[obj].r.z, obj, objects_ptr->r.x, objects_ptr->r.y, objects_ptr->r.z, dist_this );
          /* Distance less than cutoff, so calc force */
          real f_cc_this_mod = soft_sphere_force( dist_this );
          /* It is a vector force */
          vector f_cc_this;
          f_cc_this.x = f_cc_this_mod * v_dist.x / dist_this;
          f_cc_this.y = f_cc_this_mod * v_dist.y / dist_this;
          f_cc_this.z = f_cc_this_mod * v_dist.z / dist_this;
          /* Add force to f_cc */
          f_cc.x += f_cc_this.x;
          f_cc.y += f_cc_this.y;
          f_cc.z += f_cc_this.z;
          printf("C-C interaction between colloid %d and %d. Adding force %e %e %e to f_cc\n", objects_ptr->self - 10, obj, f_cc_this.x, f_cc_this.y ,f_cc_this.z );
        }
      }
    }
#ifdef NUM
    printf("Now adding to u?_new colloid %d elec force %e %e %e and cc_interactions %e %e %e\n", objects_ptr->self, 
        objects_ptr->f_ext.x, objects_ptr->f_ext.y, objects_ptr->f_ext.z, f_cc.x, f_cc.y, f_cc.z );
    ux_new = ( ux + sum_1_x + f_cc.x + objects_ptr->f_ext.x / objects_ptr->mass ) / ( 1 + sum_2_x );
    uy_new = ( uy + sum_1_y + f_cc.y + objects_ptr->f_ext.y / objects_ptr->mass ) / ( 1 + sum_2_y );
    uz_new = ( uz + sum_1_z + f_cc.z + objects_ptr->f_ext.z / objects_ptr->mass ) / ( 1 + sum_2_z );
    /* To update in case of external tau */
    wx_new = ( wx + sum_3_x ) / ( 1 + sum_4_x );
    wy_new = ( wy + sum_3_y ) / ( 1 + sum_4_y );
    wz_new = ( wz + sum_3_z ) / ( 1 + sum_4_z );
    if( fixed ){
      printf( "Particle is fixed, therefore putting u?_new %e %e %e velocity to zero now.\n", ux_new, uy_new, uz_new );
      printf( "This corresponds to imposing a %e %e %e force on colloid %d to counteract viscous + electrical forces", ux - ux_new, uy - uy_new , uz - uz_new, objects_ptr->self );
      ux_new = ux;
      uy_new = uy;
      uz_new = uz;
      wx_new = wx;
      wy_new = wy;
      wz_new = wz; 
    }
    objects_ptr->f.x = ux_new - ux;
    objects_ptr->f.y = uy_new - uy;
    objects_ptr->f.z = uz_new - uz;
    objects_ptr->t.x = wx_new - wx;
    objects_ptr->t.y = wy_new - wy;
    objects_ptr->t.z = wz_new - wz;
    
    printf("Now pushing colloid with %e %e %e\n", objects_ptr->f.x, objects_ptr->f.y, objects_ptr->f.z );
#else
    ux_new = ( ux + sum_1_x ) / ( 1 + sum_2_x );
    uy_new = ( uy + sum_1_y ) / ( 1 + sum_2_y );
    uz_new = ( uz + sum_1_z ) / ( 1 + sum_2_z );
    wx_new = ( wx + sum_3_x ) / ( 1 + sum_4_x );
    wy_new = ( wy + sum_3_y ) / ( 1 + sum_4_y );
    wz_new = ( wz + sum_3_z ) / ( 1 + sum_4_z );

    objects_ptr->f.x = ux_new + f_cc.x  + objects_ptr->f_ext.x / objects_ptr->mass - ux;
    objects_ptr->f.y = uy_new + f_cc.y  + objects_ptr->f_ext.y / objects_ptr->mass - uy;
    objects_ptr->f.z = uz_new + f_cc.z  + objects_ptr->f_ext.z / objects_ptr->mass - uz;
    /* To update in case of external tau */
    objects_ptr->t.x = wx_new - wx;
    objects_ptr->t.y = wy_new - wy;
    objects_ptr->t.z = wz_new - wz;
    if( fixed ){
      printf( "Particle is fixed, therefore putting u?_new %e %e %e velocity to zero now.\n", ux_new, uy_new, uz_new );
      printf( "This corresponds to imposing a %e %e %e force on colloid %d to counteract viscous + electrical forces", ux - ux_new, uy - uy_new , uz - uz_new, objects_ptr->self );
      ux_new = ux;
      uy_new = uy;
      uz_new = uz;
      wx_new = wx;
      wy_new = wy;
      wz_new = wz;
      objects_ptr->f.x = 0.0;
      objects_ptr->f.y = 0.0;
      objects_ptr->f.z = 0.0;
    }
#endif
   

    //printf("Velocity acquired (delta v) by particle in this timestep is %e, new vel is %e, old was %e\n", ux_new -ux, ux_new, ux );

    /* Assign bounced velocities, rule (8) JCP. Note that calculation here is simplified and works ONLY IF RHO = 24 */
    real F_bb_x = 0.0;
    real F_bb_y = 0.0;
    real F_bb_z = 0.0;
    real V_bb_x = 0.0;
    for( n_bnode = 0; n_bnode < b_nodes_ptr->num_bnode; n_bnode++ )
    {
       i_node = *( b_nodes_ptr->i_nodes + n_bnode );

       x1     = i_node >> x_shift & x_mask;
       y1     = i_node >> y_shift & y_mask;
       z1     = i_node >> z_shift & z_mask;
       l1     = i_node >> 2       & 31;
       xy1    = x1*siz_y + y1;

       x2     = x1 + x_dir[l1];
       y2     = y1 + y_dir[l1];
       z2     = z1 + z_dir[l1]; z2 = mod(z2 + params.Nz, params.Nz); // GG TODO why not PBC in X Y dirs?
       l2     = l_opp[l1];
     
       xy2    = x2 * siz_y + y2;  
       rx = *(b_nodes_ptr->r_x + n_bnode);
       ry = *(b_nodes_ptr->r_y + n_bnode);
       rz = *(b_nodes_ptr->r_z + n_bnode);

       u_node = (ux_new + wy_new*rz - wz_new*ry) * x_dir[l1] +
                (uy_new + wz_new*rx - wx_new*rz) * y_dir[l1] +
                (uz_new + wx_new*ry - wy_new*rx) * z_dir[l1] ;

       f1 = *(velcs_ptr[xy1][l1] + z1);
       f2 = *(velcs_ptr[xy2][l2] + z2);

       u_node *= weighting[l1];

       *(velcs_ptr[xy1][l1] + z1) = f2 + 4.0*u_node;
       *(velcs_ptr[xy2][l2] + z2) = f1 - 4.0*u_node;
       /* 
        * Total fluid momentum dissipated in bb collision, i.e. fluid momentum AFTER collision (sum of 2 lines above) 
        * minus fluid momentum b4 collision, f1 - f2 
        * This dissipated fluid momentum can be seen as effective force that moving colloid exterts on fluid, 
        * i.e. it is MINUS the force exerted by fluid on colloid, F_bb, so I sum the opposite sign of dissipated momentum 
        */
       F_bb_x += 2 * ( f1 - f2 - 4.0*u_node ) * x_dir[l1];
       F_bb_y += 2 * ( f1 - f2 - 4.0*u_node ) * y_dir[l1];
       F_bb_z += 2 * ( f1 - f2 - 4.0*u_node ) * z_dir[l1];

       /* Calc total normalized velocity as in Ladd part 2 eq. (3.4) */
       int velc_dir;
       real tot_px   = 0.0;
       real tot_mass = 0.0;
       for( velc_dir = 0; velc_dir < 18; velc_dir++){
         tot_mass += velcs_ptr[xy2][velc_dir][z2];
         tot_px   += velcs_ptr[xy2][velc_dir][z2] * x_dir[velc_dir];
       }
       V_bb_x += tot_px / tot_mass;
    }
    printf("F frictional from fluid to colloid %d is %e %e %e\n", objects_ptr->self, F_bb_x, F_bb_y, F_bb_z );
    //real avgVx = V_bb_x / b_nodes_ptr->num_bnode;
    //printf("Total velocity_x around colloid is %e -> AVG vel over surface (num_bnode) is %e, so xi effective is %e\n", V_bb_x, avgVx, F_bb_x / -avgVx);
}

real soft_sphere_force( real h )
{
  real f = 0.0;
  real hc = 0.5;
  real nu = 2.0;
  real epsilon = 1.0;
  real sigma = 0.5;

  if( h > 0 && h < hc ){
    f = pow( h, -(nu+1)) - pow(hc, -(nu+1) );
    f = f * epsilon * pow( sigma, nu ) * nu;
  }
  return f;
}

int update( object * objects_ptr, b_node * bnodes_ptr, int ** site_type, real ** c_plus, real ** c_minus, real *** velcs_ptr )
{
    int moved = 0;

    int ctr_i_old, ctr_j_old, ctr_k_old;
    ctr_i_old = objects_ptr->i.x;
    ctr_j_old = objects_ptr->i.y;
    ctr_k_old = objects_ptr->i.z;

    real rx, ry, rz;
    rx = objects_ptr->r.x + objects_ptr->u.x;
    ry = objects_ptr->r.y + objects_ptr->u.y;
    rz = objects_ptr->r.z + objects_ptr->u.z;
    if( rx < 0.5 )
      rx += (real) params.Nx;
    if( rx > params.Nx + 0.5 )
      rx -= (real) params.Nx;
    if( ry < 0.5 )
      ry += (real) params.Ny;
    if( ry > params.Ny + 0.5 )
      ry -= (real) params.Ny;
    if( rz < -0.5 )
      rz += (real) params.Nz;
    if( rz > params.Nz - 0.5 )
      rz -= (real) params.Nz;

    objects_ptr->i.x  =  nint2( objects_ptr->r.x );
    objects_ptr->i.y  =  nint2( objects_ptr->r.y );
    objects_ptr->i.z  =  nint2( objects_ptr->r.z );

    objects_ptr->r.x = rx;
    objects_ptr->r.y = ry;
    objects_ptr->r.z = rz;
    /*
     * WE DO NOT ADD EXTERNAL FORCE HERE AS ALREADY ADDED IN THE EXPLICIT CALCULATION OF u_NEW, 
     * i.e. EFFECT OF EXT FORCE AND BB FORCE ARE BOTH IN objects_ptr->f.x;

     * Updating colloid velocity. New vel was calculated with BB + force, here I just apply
     * the force that gives that new velocity from the old one
     */
    objects_ptr->u.x += objects_ptr->f.x;
    objects_ptr->u.y += objects_ptr->f.y;
    objects_ptr->u.z += objects_ptr->f.z;
    objects_ptr->w.x += objects_ptr->t.x;
    objects_ptr->w.y += objects_ptr->t.y;
    objects_ptr->w.z += objects_ptr->t.z;
    
  /* If colloid DISCRETELY moved */
  if( params.move && (ctr_i_old != objects_ptr->i.x || ctr_j_old != objects_ptr->i.y || ctr_k_old != objects_ptr->i.z) ){
    vector r_jump;
    /* Work out the jump versor */
    pbc_vec_dist( objects_ptr->i.x, objects_ptr->i.y, objects_ptr->i.z, ctr_i_old, ctr_j_old, ctr_k_old, &r_jump.x, &r_jump.y, &r_jump.z );

    printf("Jump (colloid %d) of %f %f %f\n", objects_ptr->self, r_jump.x, r_jump.y, r_jump.z );
    /* Looking for SOLID to FLUID nodes and their twins (F->S) */
    int i, j, k;
    //int l;
    /* Not possible to loop over size_[mp] as colloid move and can be on the box edge */
    for( i = 1 ;i <= params.Nx ; i++ ){
      for( j = 1 ;j <= params.Ny ; j++ ){
	        int index = i * ( params.Ny + 2 ) + j;
	        for( k = 0; k < params.Nz; k++ ){
          /* Check SOLID nodes of THIS particular colloid! */
          if( site_type[index][k] == objects_ptr->self ){
            /* Work out if this SOLID node has to disappear, if not continue */
            real dist = pbc_dist(i, j, k, objects_ptr->i.x, objects_ptr->i.y, objects_ptr->i.z);
            if( dist <= objects_ptr->rad ) continue;
            printf( "Node %d %d %d was solid and now it is %f from center of colloid %d\n", i, j, k, dist, objects_ptr->self );
            int twin_i, twin_j, twin_k;
            find_twin( i, j, k, objects_ptr, r_jump, &twin_i, &twin_j, &twin_k );
            /* Make sure twin is still in box, i.e. PBC */
            //pbc_pos( &twin_i, &twin_j, &twin_k );
            /* Swapping fluid and charges between twins */
            //real node_fluid[18];
            real node_ch_plus;
            real node_ch_minus;
            int twin_index = twin_i * ( params.Ny + 2 ) + twin_j;
            int twin_twin_i = twin_i + r_jump.x;
            int twin_twin_j = twin_j + r_jump.y;
            int twin_twin_k = twin_k + r_jump.z;
            pbc_pos( &twin_twin_i, &twin_twin_j, &twin_twin_k );
            int twin_twin_index = twin_twin_i * ( params.Ny + 2 ) + twin_twin_j;
            /* CP oldSOLID to back up node */
            node_ch_plus  = c_plus[index][k];
            node_ch_minus = c_minus[index][k];
            //for( l = 0; l < 18; l++ ) node_fluid[l] = velcs_ptr[index][l][k];
            /* CP twin_twin to old SOLID (that will become newFLUID) */
            c_plus[index][k]  = c_plus[twin_twin_index][twin_twin_k];
            c_minus[index][k] = c_minus[twin_twin_index][twin_twin_k];
            //for( l = 0; l < 18; l++ )  velcs_ptr[index][l][k] = velcs_ptr[twin_index][l][twin_k];
            /* CP twin to twin_twin */
            c_plus[twin_twin_index][twin_twin_k]  = c_plus[twin_index][twin_k];
            c_minus[twin_twin_index][twin_twin_k] = c_minus[twin_index][twin_k];
            /* CP oldSOLID to twin (via backup) */
            c_plus[twin_index][twin_k] = node_ch_plus;
            c_minus[twin_index][twin_k] = node_ch_minus;
            //for( l = 0; l < 18; l++ )  velcs_ptr[twin_index][l][twin_k] = node_fluid[l];
            /* bnodes_init deals only with SOLID NODES -> I put disappearing SOLID to FLUID */
            site_type[index][k] = FLUID;
          }
        }
      }
    }
    bnodes_init( objects_ptr, bnodes_ptr, objects_ptr->self, site_type );
    moved = 1;
  }
  return moved;
}

inline real rpbc_vec_dist( real i, real j, real k, real a, real b, real c, vector * v_dist )
{
  real hNx = params.Nx / 2.0;
  real hNy = params.Ny / 2.0;
  real hNz = params.Nz / 2.0;
  real dist_x, dist_y, dist_z;
  dist_x = i - a;
  dist_y = j - b;
  dist_z = k - c;
  if( dist_x > hNx )  dist_x -= params.Nx;
  if( dist_y > hNy )  dist_y -= params.Ny;
  if( dist_z > hNz )  dist_z -= params.Nz;
  if( dist_x < -hNx ) dist_x += params.Nx;
  if( dist_y < -hNy ) dist_y += params.Ny;
  if( dist_z < -hNz ) dist_z += params.Nz;
  v_dist->x = dist_x;
  v_dist->y = dist_y;
  v_dist->z = dist_z;
  real dist = pow( dist_x * dist_x + dist_x * dist_y + dist_z * dist_z, 0.5 );
  return dist;
}

inline void pbc_vec_dist( int i, int j, int k, int a, int b, int c, real * dist_x, real * dist_y, real * dist_z )
{
  real hNx = params.Nx / 2.0;
  real hNy = params.Ny / 2.0;
  real hNz = params.Nz / 2.0;
  *dist_x = i - a;
  *dist_y = j - b;
  *dist_z = k - c;
  if( *dist_x > hNx ) *dist_x -= params.Nx;
  if( *dist_y > hNy ) *dist_y -= params.Ny;
  if( *dist_z > hNz ) *dist_z -= params.Nz;
  if( *dist_x < -hNx ) *dist_x += params.Nx;
  if( *dist_y < -hNy ) *dist_y += params.Ny;
  if( *dist_z < -hNz ) *dist_z += params.Nz;
}

inline real pbc_dist( int i, int j, int k, int a, int b, int c )
{
  real dist_x, dist_y, dist_z, dist_r;
  real hNx = params.Nx / 2.0;
  real hNy = params.Ny / 2.0;
  real hNz = params.Nz / 2.0;
  dist_x = i - a;
  dist_y = j - b;
  dist_z = k - c;
  if( dist_x > hNx ) dist_x -= params.Nx;
  if( dist_y > hNy ) dist_y -= params.Ny;
  if( dist_z > hNz ) dist_z -= params.Nz;
  if( dist_x < -hNx ) dist_x += params.Nx;
  if( dist_y < -hNy ) dist_y += params.Ny;
  if( dist_z < -hNz ) dist_z += params.Nz;
  dist_r = sqrt( dist_x * dist_x + dist_y * dist_y + dist_z * dist_z );
  return dist_r;
}

void pbc_pos( int * x, int * y, int * z )
{
  if( *x > params.Nx ) *x -= params.Nx;
  if( *y > params.Ny ) *y -= params.Ny;
  if( *z > params.Nz ) *z -= params.Nz;
  if( *x < 1 ) *x += params.Nx;
  if( *y < 1 ) *y += params.Ny;
  if( *z < 0 ) *z += params.Nz;
  //printf(" and now %f %f %f\n", *x, *y, *z );
}
    

void find_twin( int i, int j, int k, object * objects_ptr, vector r_jump, int * twin_i, int * twin_j, int * twin_k )
{
  /* Check r_jump magnitude is ok, i.e. only a jump of 1 in modulus (no 2 directions at once) */
  if( r_jump.x * r_jump.x + r_jump.y * r_jump.y * r_jump.z * r_jump.z > 1.0 ){
    printf( "PROBLEM! Colloid is trying to jump in a diagonal direction! Not implemented. Bye!\n" );
    exit( 0 );
  }

  int ctr_i_old = (int) (objects_ptr->i.x - r_jump.x);
  int ctr_j_old = (int) (objects_ptr->i.y - r_jump.y);
  int ctr_k_old = (int) (objects_ptr->i.z - r_jump.z);
  vector delta;
  delta.x = i - ctr_i_old;
  delta.y = j - ctr_j_old;
  delta.z = k - ctr_k_old;  
  
  *twin_i = (int) (i - 2.0 * delta.x + r_jump.x);
  *twin_j = (int) (j - 2.0 * delta.y + r_jump.y);
  *twin_k = (int) (k - 2.0 * delta.z + r_jump.z);
  pbc_pos( twin_i, twin_j, twin_k );
  printf("Colloid moving along %f %f %f. Remove SOLID at %d %d %d to appear at %d %d %d\n", r_jump.x, r_jump.y, r_jump.z, i, j, k, *twin_i, *twin_j, *twin_k );
}
