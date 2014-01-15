#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "simpar.h"
#include "lb.h"
#include "init.h"
#include "mc_equilibration.h"

extern sim_params_t params;

extern long int idum;
extern int x_dir[18];
extern int y_dir[18];
extern int z_dir[18];
extern int x_shift;
extern int y_shift;
extern int z_shift;

int fluid_nodes;

#define PIG 3.1415927

real *** alloc_3mtx( int n )
{
  /* vels[Lxy][18][max_z] for velcs_ptr */
  /* vels[Lxy][3][max_z]  for solute */
  /* For some reasons, Lxy as below. Not touching this for now, as could be important, ex. PBC */
  int i, j;
  int Lxy = ( params.Nx + 2 ) * ( params.Ny + 2 );

  /* Create matrix of * pointers */
  real ***  ppp  = (real ***)  malloc( (size_t) Lxy *      sizeof(real **) );
  real **   pp   = (real  **)  malloc( (size_t) Lxy * n * sizeof(real *) );
  for( i = 0; i < Lxy; i++ ){
    ppp[i] = (real **) pp + i * n;
  }
  /* Create space for 3D matrix values */
  real * a = (real *) calloc( Lxy * n * params.Nz, sizeof(real) );
  /* Initialize values of matrix pointers to addresses in space of values */
  for( i = 0; i < Lxy; i++ ){
    for( j = 0; j < n; j++ ){
      ppp[i][j] = (real *) a + (i * n + j) * params.Nz;
    }
  }
  return ppp;
}

/*************************************************************/
real ** mem_r_lattice2( int max_index, int max_z )
/*************************************************************/
{
  real ** pp = (real **) malloc( (size_t) max_index * sizeof(real *) );
  real *  a  = (real *)  calloc( max_index * max_z, sizeof(real) );
  if(a == NULL){
    printf("mem_r_lattice2 failed! Bye\n");
    exit( 0 );
  }
  int i;
  for( i = 0; i < max_index; i++ ){
    pp[i] = (real *) a + i * max_z;
  }
  //printf( "pp @ %p and pp is %p pp[0] @ %p pp[1] @ %p a @ %p\n", pp, &pp, pp[0], pp[1], a );
  return pp;
}

/*************************************************************/
int ** mem_i_lattice2( int max_index, int max_z )
/*************************************************************/
{
  int ** pp = (int **) malloc( (size_t) max_index * sizeof(int *) );
  int * a = (int *) calloc( max_index * max_z, sizeof(int) );
  int i;
  for( i = 0; i < max_index; i++ ){
    pp[i] = (int *) a + i * max_z;
  }
  return pp;
}

/*************************************************************/
real ** mem_r_lattice( int max_index, int max_z )
/*************************************************************/
{
  real ** p = (real **) malloc( (size_t) max_index * sizeof(real *) );
  int i;
  for(i = 0; i < max_index; i++){
    p[i] = (real *)calloc( max_z, sizeof(real) );
  }
  return p;
}

/*************************************************************/
int ** mem_i_lattice( int max_index, int max_z )
/*************************************************************/
{
  int ** p = (int **) malloc( (size_t) max_index * sizeof(int *) );
  int i;
  for(i = 0; i < max_index; i++){
    p[i] = (int *)calloc( max_z, sizeof(int) );
  }
  return p;
}

void free_2D( void ** ptr )
{
  free( ptr[0] );
  free( ptr );
}

void free_3D( void *** ptr )
{
  free( ptr[0][0] );
  free( ptr[0] );
  free( ptr );
}


/*************************************************************/
void init_GK( real *** velcs_ptr, object * obj, real ** c_plus, real ** c_minus, int ** site_type, real ** rho, real ** jx, real ** jy, real ** jz )
/*************************************************************/
{
  int i, j, k;
  int siz_y = params.Ny + 2;
  real E = (real) params.GreenKubo / 100.0 ;
  printf( "GREENKUBO: Perturbing field is %e\n", E );
  /* 
   * Prepare external delta perturbation of charges
   * Ext perturbation on each node is (see equations) \dot{p_i} = e * E * charge_i
   * Implement change of momentum in fluid nodes using solute_force.
   * Depending on the kicking setup (see also below), momentum can be added
   * to the fluid of colloid nodes (factor = 1.0) or to the CoM 
   * of particle setting obj->u.x (factor = 0.0) */
  real * solute_force[3];
  vector ptot, str;
  solute_force[0] = (real *)calloc(  params.Nz, sizeof( real ) );
  solute_force[1] = (real *)calloc(  params.Nz, sizeof( real ) );
  solute_force[2] = (real *)calloc(  params.Nz, sizeof( real ) );
  real tot_px, tot_py, tot_pz;
  tot_px = tot_py = tot_pz = 0.0;
  for( i = 1; i <= params.Nx; i++){
    for( j = 1; j <= params.Ny; j++ ){
      int index = i * ( params.Ny + 2 ) + j;
      /* Prepare solute_force for this column */
    	for( k = 0; k < params.Nz; k++ ){
        solute_force[0][k] =  solute_force[1][k]  =  solute_force[2][k] = 0.0;
	      if( site_type[index][k] == FLUID ){
          solute_force[0][k] = E * ( c_plus[index][k] - c_minus[index][k] );
          solute_force[1][k] = E * 2 * ( c_plus[index][k] - c_minus[index][k] );
          solute_force[2][k] = E * 3 * ( c_plus[index][k] - c_minus[index][k] );
        }else{
          //real factor = 1.0;
          real factor = 0.0;
          printf( "GREENKUBO: factor is equil to %f\n", factor );
          solute_force[0][k] = factor * E * ( c_plus[index][k] - c_minus[index][k] );
          solute_force[1][k] = factor * 2 * E * ( c_plus[index][k] - c_minus[index][k] );
          solute_force[2][k] = factor * 3 * E * ( c_plus[index][k] - c_minus[index][k] );
        }
      }
      /* Insert momentum letting fluid particles collide */
	    do_z_column( velcs_ptr[index], params.nu, i, j, &ptot, &str, solute_force, 
        jx[index], jy[index], jz[index], rho[index], site_type[index] );
      /* Keep track of total momentum inserted so to use it to kick colloid */
    	for( k = 0; k < params.Nz; k++ ){ 
        int velc_dir;
        for( velc_dir = 0; velc_dir < 18; velc_dir++ ){
	        if( site_type[index][k] == FLUID ){
            tot_px += velcs_ptr[index][velc_dir][k] * x_dir[velc_dir];
            tot_py += velcs_ptr[index][velc_dir][k] * y_dir[velc_dir];
            tot_pz += velcs_ptr[index][velc_dir][k] * z_dir[velc_dir];
          }
        }
	    }
    }
  }
  real th_px = - E * params.ch_objects;
  printf( "GREENKUBO: Reloaded (zero) momentum and perturbed fluid. MOM TOT (fluid only nodes) X %e Y %e Z %e. Theoretically should be %e %e %e\n", 
      tot_px, tot_py, tot_pz, th_px, 2 * th_px, 3 * th_px );
  /* Now perturb colloid. */
  /* I can also perturb colloid adding momentum directly to the center of mass
   * In this case put factor above = 0.0, so fluid inside colloid has zero velocity
   * but uncomment line below so that colloid is kicked */
  obj->u.x = params.kbt * E * params.ch_objects / obj->mass;
  obj->u.y = params.kbt * 2 * E * params.ch_objects / obj->mass;
  obj->u.z = params.kbt * 3 * E * params.ch_objects / obj->mass;
  printf( "GREENKUBO: Particle velocity (see factor above) only due to kick is %e %e %e\n", obj->u.x, obj->u.y, obj->u.z );

  /* Final check total momentum (should be zero) */
  tot_px = tot_py = tot_pz = 0.0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        int velc_dir;
        for( velc_dir = 0; velc_dir < 18; velc_dir++ ){
          tot_px += velcs_ptr[index][velc_dir][k] * x_dir[velc_dir];
          tot_py += velcs_ptr[index][velc_dir][k] * y_dir[velc_dir];
          tot_pz += velcs_ptr[index][velc_dir][k] * z_dir[velc_dir];
        }
      }
    }
  }
  printf( "GREENKUBO: TOTAL momentum end GK init. X %e Y %e Z %e\n", tot_px, tot_py, tot_pz );
  free( solute_force[0] );
  free( solute_force[1] );
  free( solute_force[2] );
}

/*************************************************************/
void init_wall( int ** site_type )
/*************************************************************/
{
  /* initialization of charged wall */
  /* the geometry assumes that there are 2 solid walls. Each wall has a
  width "width". If width is NEGATIVE, there are no walls */
  /* if there is a wall, its charge is specified and the wall charge is 
  distributed as a volume charge over all the solid nodes. 
  This charge density is quenched and does not evolve dynamically.
  For the time being the walls are located at the end of the system in the 
  z direction.*/
  int i, j, k;
  int siz_y = params.Ny + 2;

  for(i = 0; i <= params.Nx; i++){
    for(j = 0; j <= params.Ny; j++){
      int index = i * siz_y + j;
      for(k = 0; k < params.Nz; k++){
	      if( (k <= params.width) || (k>= params.Nz - params.width - 1) ){
	        /* inside is a flag that identifies solid nodes */
	        site_type[index][k] = WALL;
	      }else{
	        site_type[index][k] = FLUID;
	      }
      }
    }
  }
}



/*************************************************************/
void      conf_gen(  object *objects, int *n_obj, real rsq)
/*************************************************************/
{
  int i, j, imax, flag;
  real rx_val, ry_val, rz_val;
  real *rx, *ry, *rz, r;

  real max_x    = params.Nx;
  real max_y    = params.Ny;
  real max_z    = params.Nz;

  rsq = rsq*rsq;
  
  if (*n_obj <2){
    /* for one particle: you place it here in the system */
# if DIM3 == 1
    objects[0].r.x = (real) max_x*0.5;
# else
    objects[0].r.x = 1.0;
# endif
    objects[0].r.y = (real) (max_y)/2.;
    objects[0].r.z = (real) (max_z)/2.;
    objects[0].i.x = (int) objects[0].r.x;
    objects[0].i.y = (int) objects[0].r.y;
    objects[0].i.z = (int) objects[0].r.z;
    objects[0].rad = sqrt(rsq);
  }else{
    rx = (real *)malloc( (*n_obj)*sizeof(real));
    ry = (real *)malloc( (*n_obj)*sizeof(real));
    rz = (real *)malloc( (*n_obj)*sizeof(real));
    for ( i = 0; i<(*n_obj); i++){
      flag = TRUE;
# if DIM3 == 1
      rx[i] =  0.5 + max_x*ran1( (long *) &idum);
# else
      rx[i] = 1.0;
# endif
      ry[i] =  0.5 + max_y*ran1( (long *) &idum);
      rz[i] = -0.5 + max_z*ran1( (long *) &idum);
      rx[0]=max_x/2;
      ry[0]=max_y/2-sqrt(rsq);
      rz[0]=max_z/2;
      rx[1]=max_x/2;
      ry[1]=max_y/2+sqrt(rsq);
      rz[1]=max_z/2;
      imax = i;

      for(j = 0; j < imax; j++){
# if DIM3 == 1
	rx_val = fabs(rx[i] - rx[j]);
	rx_val = min(rx_val, fabs(max_x-rx_val));
# else
	rx_val =0.0;
# endif
	ry_val = fabs(ry[i] - ry[j]);
	ry_val=min(ry_val,fabs(max_y-ry_val));
	rz_val = rz[i] - rz[j];
	rz_val=min(rz_val,fabs(max_z-rz_val));
	r = rx_val*rx_val + ry_val*ry_val +rz_val*rz_val;
	if ( r < 4.0*rsq){
	  j = i;
	  i--;
	  imax = 0;
	}
      }
      if (imax == i){
	objects[i].r.x =  rx[i];
	objects[i].r.y =  ry[i];
	objects[i].r.z =  rz[i];
	objects[i].i.x = nint( rx[i]);
	objects[i].i.y = nint( ry[i]);
	objects[i].i.z = nint( rz[i]);
	objects[i].rad = sqrt(rsq);
	
	printf("la particula i %d %d %e %e %e\n", i, *n_obj, objects[i].r.x,objects[i].r.y,objects[i].r.z);
	
      }
    }
    free(rx);
    free(ry);
    free(rz);
  }
}

/*************************************************************/
void  conf_gen2(  object *objects, real r_eff, real r_pair )
/*************************************************************/
{
  int i;
  real cx[2], cy[2], cz[2];
  
  /* Work out max and min r_pair depending on lattice and check with input r_pair and r_eff */
  // GG here parts can be placed at 3 different distances in x,y,z so to save simulations, 
  // but now keep it simple and just use direction x
  real max_r_dist = params.Nx / 2.0 - 2.0 * r_eff;
  if( r_pair >= max_r_dist ){
    printf("Uh! Uh! r_eff or Lx cannot be used for the chosen r_pair, bye!\n"); 
    exit( 1 );
  }

  /* Calculate vanilla lattice nodes where to put colloids */
  cx[0] = params.Nx / 4.0;
  cx[1] = params.Nx / 4.0 + 2.0 * r_eff + r_pair;
  cy[0] = cy[1] = params.Ny / 2.0;
  cz[0] = cz[1] = params.Nz / 2.0;
  
  /* Place colloids */
  for( i = 0; i < 2; i++ ){
	  objects[i].i.x = nint( cx[i]);
	  objects[i].i.y = nint( cy[i]);
	  objects[i].i.z = nint( cz[i]);
    /* Note here that I force continuos center to coincide with integer center
     * so there is not misunderstanding between continuos center position and 
     * center of mass (which is used to define link nodes r?)
     */
	  objects[i].r.x = objects[i].i.x;
	  objects[i].r.y = objects[i].i.y;
	  objects[i].r.z = objects[i].i.z;
    objects[i].rad = r_eff;
    printf( "\tPlaced COLLOID %d in %f %f %f\n", i, cx[i], cy[i], cz[i] );
  }
}


int conf_fromfile(  object *objects, real r_eff )
{
  int neg_colls = 0;
  float cx, cy, cz, q;
  /* Scan input file */
  FILE * infile = fopen( params.inf_colloids, "r" );
        //fprintf( outfile, "%d\t%d\t%f\t%f\t%f\t%f\n", step, p, parts[p].x, parts[p].y, parts[p].z, energy );
  /* Place colloids */
  int i;
  for( i = 0; i < params.num_obj; i++ ){
    fscanf( infile, "%f %f %f %f\n", &cx, &cy, &cz, &q );
	  objects[i].r.x = cx;
	  objects[i].r.y = cy;
	  objects[i].r.z = cz;
	  objects[i].i.x = nint( cx );
	  objects[i].i.y = nint( cy );
	  objects[i].i.z = nint( cz );
	  objects[i].rad = r_eff;
	  objects[i].charge = q;
    if( q < 0.0 ) neg_colls++;
    printf( "\tPlaced COLLOID %d with charge %f in %f %f %f INT (%d %d %d)\n", i, q, cx, cy, cz, objects[i].i.x, objects[i].i.y, objects[i].i.z );
  }
  fclose( infile );
  return neg_colls;
}

/*************************************************************/
void           objects_init( object  *objects_ptr, int n_obj)
/*************************************************************/
{
    
    objects_ptr->self        =  SOLID + n_obj;
    objects_ptr->mass_flag   =  1;
    objects_ptr->mass_fac    =  1.0; 
#   if DIM3 == 1    
    objects_ptr->size_m.x    =  -objects_ptr->rad - 1;  /* Object */
    objects_ptr->size_p.x    =  objects_ptr->rad + 1;   /* Object */
#   else						/* Object */
    objects_ptr->size_m.x    =  0;			/* Object */
    objects_ptr->size_p.x    =  0;			/* Object */
#   endif    						/* Object */
    objects_ptr->size_m.y    =  (int) -objects_ptr->rad - 1;  /* Object */
    objects_ptr->size_m.z    =  (int) -objects_ptr->rad - 1;  /* sizes  */ 
    objects_ptr->size_p.y    =  (int) objects_ptr->rad + 1;
    objects_ptr->size_p.z    =  (int) objects_ptr->rad + 1;
   
    objects_ptr->i.x         =  nint2( objects_ptr->r.x );
    objects_ptr->i.y         =  nint2( objects_ptr->r.y );
    objects_ptr->i.z         =  nint2( objects_ptr->r.z );
    objects_ptr->max_bnode   =  0;
    objects_ptr->f.x         =  0.0;  /* Forces  */    
    objects_ptr->f.y         =  0.0;
    objects_ptr->f.z         =  0.0;
    objects_ptr->t.x         =  0.0;  /* Torques */
    objects_ptr->t.y         =  0.0;
    objects_ptr->t.z         =  0.0;
    objects_ptr->u.x         =  0.0;

    objects_ptr->u.y         =  0.00*ran1( (long *) &idum );
    objects_ptr->u.z         =  0.00*ran1( (long *) &idum );

    objects_ptr->w.x         =  0.0;
    objects_ptr->w.y         =  0.0;

    objects_ptr->w.z         =  0.0;
    objects_ptr->f_ext.x     =  0.0;  /* External Forces */
    objects_ptr->f_ext.y     =  0.0;
    objects_ptr->f_ext.z     =  0.0;
    objects_ptr->t_ext.x     =  0.0;  /* External Torques */
    objects_ptr->t_ext.y     =  0.0;
    objects_ptr->t_ext.z     =  0.0;
    objects_ptr->theta       =  0.0;
    objects_ptr->r_sq 	     = objects_ptr->rad*objects_ptr->rad;
/*		3D	OBJECTS		*/
#   if    DIM3 == 1	
      objects_ptr->mass        =  objects_ptr->r_sq*objects_ptr->rad;
      objects_ptr->mass        *= objects_ptr->mass_fac*100.53;
      objects_ptr->inertia     = 0.4*objects_ptr->mass;
      objects_ptr->inertia     *= objects_ptr->r_sq;
#   else  
/*		2D	OBJECTS		*/
      objects_ptr->mass        =  objects_ptr->r_sq;
      objects_ptr->mass        *= objects_ptr->mass_fac*75.398224;
      objects_ptr->inertia     = 0.5*objects_ptr->mass;
      objects_ptr->inertia     *= objects_ptr->r_sq;
#   endif   
}

/*************************************************************/
void bnodes_init( object *objects_ptr, b_node *bnodes_ptr, int coll_label, int *site_type[] )
/*************************************************************/
{
  int x, y, z, velc_dir, index, index_2;
  int x_abs, y_abs, z_abs;
  int x_abs_pbc, y_abs_pbc, z_abs_pbc;
  int x_c, y_c, z_c;
  int x_c_pbc, y_c_pbc, z_c_pbc;
  real x_real, y_real, z_real;
  real rx, ry, rz;
  int siz_y = params.Ny + 2;

  //printf("%p %p %p %p\n", bnodes_ptr->i_nodes, bnodes_ptr->r_x, bnodes_ptr->r_y, bnodes_ptr->r_z );
  free( bnodes_ptr->i_nodes );
  free( bnodes_ptr->r_x );
  free( bnodes_ptr->r_y );
  free( bnodes_ptr->r_z );
  bnodes_ptr->i_nodes = (int *)    malloc( 1 * sizeof(int) );
  bnodes_ptr->r_x     = (real *)  malloc( 1 * sizeof( real) );
  bnodes_ptr->r_y     = (real *)  malloc( 1 * sizeof(real) );
  bnodes_ptr->r_z     = (real *)  malloc( 1 * sizeof(real) );
  bnodes_ptr->num_int_nodes = bnodes_ptr->num_bnode = 0;
   
  
  x_real  = (float) (objects_ptr->i.x);
  y_real  = (float) (objects_ptr->i.y);
  z_real  = (float) (objects_ptr->i.z);
  
  int num_solid = 0;
  for( x = objects_ptr->size_m.x; x <= objects_ptr->size_p.x; x++){ 
    for( y = objects_ptr->size_m.y; y <= objects_ptr->size_p.y; y++){ 
      for( z = objects_ptr->size_m.z; z <= objects_ptr->size_p.z; z++){
        x_abs = x + objects_ptr->i.x;
	      y_abs = y + objects_ptr->i.y;
	      z_abs = z + objects_ptr->i.z;
	      x_abs_pbc = mod( x_abs + params.Nx - 1, params.Nx ) + 1; 
	      y_abs_pbc = mod( y_abs + params.Ny - 1, params.Ny ) + 1; 
	      z_abs_pbc = mod( z_abs + params.Nz, params.Nz );
	      index = x_abs_pbc * siz_y + y_abs_pbc;
/* 3D BOUNDARY */
#if DIM3 == 1
	      if( disk( (real)x_abs - x_real, (real)y_abs - y_real, (real)z_abs - z_real,  objects_ptr ) == 1 )
#else                         
/* 2D BOUNDARY */ 
	      if( disk2d((real)y_abs - y_real, (real)z_abs - z_real,  objects_ptr ) == 1 )
#endif             
	      { /* IF SITE IS IN OBJECT */
	        /* We're in solid */             
          num_solid++;
		      site_type[index][z_abs_pbc] = coll_label;

	        for( velc_dir = 0; velc_dir < 18; velc_dir++){
		        x_c = x_abs + x_dir[velc_dir];
		        y_c = y_abs + y_dir[velc_dir];
		        z_c = z_abs + z_dir[velc_dir];
		        x_c_pbc = mod(x_c + params.Nx -1, params.Nx) + 1; 
		        y_c_pbc = mod(y_c + params.Ny -1, params.Ny) + 1; 
		        z_c_pbc = mod(z_c + params.Nz, params.Nz);
		        index_2 = x_c_pbc*siz_y + y_c_pbc;
/* 3D BOUNDARY */
#if DIM3 == 1
		        if( disk( (real)x_c - x_real, (real)y_c - y_real, (real)z_c - z_real,  objects_ptr ) == 0 )
#else                       
/* 2D BOUNDARY */                           
		        if( disk2d((real)y_c - y_real, (real)z_c - z_real,  objects_ptr ) == 0 )
#endif                              
		        { /* IF N.N. velc_dir IS NOT IN OBJECT */
			        rx = 0.5 * ( x_abs + x_c - 2.0 * x_real );
			        ry = 0.5 * ( y_abs + y_c - 2.0 * y_real );
			        rz = 0.5 * ( z_abs + z_c - 2.0 * z_real );
			        (bnodes_ptr->i_nodes ) = (int *)   realloc( bnodes_ptr->i_nodes, (bnodes_ptr->num_bnode + 1) * sizeof(int) );
			        (bnodes_ptr->r_x )     = (real *) realloc( bnodes_ptr->r_x,     (bnodes_ptr->num_bnode + 1) * sizeof(real) );
			        (bnodes_ptr->r_y )     = (real *) realloc( bnodes_ptr->r_y,     (bnodes_ptr->num_bnode + 1) * sizeof(real) );
			        (bnodes_ptr->r_z )     = (real *) realloc( bnodes_ptr->r_z,     (bnodes_ptr->num_bnode + 1) * sizeof(real) );

              *(bnodes_ptr->i_nodes + bnodes_ptr->num_bnode) = x_abs_pbc << x_shift | y_abs_pbc << y_shift | z_abs_pbc << z_shift | velc_dir << 2 | 0; 
			        *(bnodes_ptr->r_x + bnodes_ptr->num_bnode) = rx; 
			        *(bnodes_ptr->r_y + bnodes_ptr->num_bnode) = ry;
			        *(bnodes_ptr->r_z + bnodes_ptr->num_bnode) = rz;
              //printf("Bnode %d in %d %d %d with fluid nn in %d %d %d rx %f ry %f rz %f\n", bnodes_ptr->num_bnode, x_abs, y_abs, z_abs, x_c, y_c, z_c, rx, ry, rz );
			        bnodes_ptr->num_bnode++;
		        } /* END IF N.N. velc_dir IS NOT IN OBJECT */
		      }  /* END LOOPING VELC_DIR */
	      } /* END IF SITE IS IN OBJECT */
      } 
    }         
  } 
  objects_ptr->n_points = num_solid;
  //printf("Bnodes built colloid %d in %d %d %d with %d solid sites\n", objects_ptr->self, objects_ptr->i.x, objects_ptr->i.y, objects_ptr->i.z, objects_ptr->n_points );
}

/*************************************************************/
int disk( real x, real y, real z, object *objects_ptr  )
/*************************************************************/
{
   if( (x*x+y*y +z*z) <= objects_ptr->r_sq ){
       return (1);
   }else{
       return (0);
   } 
}
/*************************************************************/
int disk2d(real y, real z, object *objects_ptr  )
/*************************************************************/
{
   if( (y*y +z*z) <= objects_ptr->r_sq ){
       return (1);
   }else{
       return (0);
   } 
}


real anormf0;
/*************************************************************/
void init_ch_dist(int lx,int ly, int lz, int ** site_type, int  totsit, double **c_minus, double **c_plus, real ** el_potential, object *objects)
/*************************************************************/
{
  real objects_charge = params.ch_objects * params.num_obj;
  real wall_charge = params.ch_wall;  
  int siz_y = params.Ny + 2;

  /* Count solid wall and fluid sites */
  int i, j, k, index;
  int tot_sites, obj_sites, wall_sites;
  tot_sites = 0; obj_sites = 0; wall_sites = 0;
  for( i = 1; i <= lx; i++ ){
    for( j = 1; j <= ly; j++ ){
      index = siz_y * i + j;
      for( k = 0; k < lz; k++ ){
        if( site_type[index][k] >= SOLID ) obj_sites++;
        if( site_type[index][k] == WALL  ) wall_sites++; 
        tot_sites++;
      }
    }
  }
  fluid_nodes = tot_sites - wall_sites - obj_sites;
  printf( "\nLattice of %d sites:\n\t%d wall sites (width %d)\n\t%d object sites\n\t%d fluid sites\n", tot_sites, wall_sites, params.width, obj_sites, fluid_nodes );

  /* Work out charge density per wall/solid/fluid site */ 
  real rho_wall, rho_solid;
  real rho_p_solid = 0.0; 
  real rho_m_solid = 0.0;
  real rho_p_wall  = 0.0;
  real rho_m_wall  = 0.0;
  if( obj_sites ){
    rho_solid = objects_charge / (real) obj_sites;
    rho_p_solid = 0.5 * rho_solid; 
    rho_m_solid = -rho_p_solid;
  }
  if( wall_sites ){
    rho_wall  = wall_charge  / (real) wall_sites;
    rho_p_wall  = 0.5 * rho_wall; 
    rho_m_wall  = -rho_p_wall;
  }

  /* Calculate rho_fluid and change sign to it to guarantee electroneutrality */
  /* Charge on fluid is equal to the charge on obj (if objects) plus charge on wall (if wall) */
  real rho_fluid  = ( params.num_obj ) ? ( objects_charge ) : 0;
  rho_fluid += ( wall_sites ) ? ( wall_charge ) : 0;

  rho_fluid = ( rho_fluid ) / (real) fluid_nodes;
  rho_fluid = -rho_fluid;
  /* Add salt to fluid */
  real con_salt = 1.0 / ( 4 * PIG * params.ch_bjl * params.ch_lambda * params.ch_lambda );

  if( params.ch_lambda == 0.0 ){  
    con_salt = 0.0;
    printf("\t*****   THIS IS A NO SALT SIMULATION     *****\n");
  }

  real rho_p_fluid, rho_m_fluid;
  if( rho_fluid >= 0.0){
    rho_p_fluid = rho_fluid + 0.5 * con_salt; 
    rho_m_fluid = 0.5 * con_salt; 
  }else{
    rho_m_fluid = 0.5 * con_salt - rho_fluid; 
    rho_p_fluid = 0.5 * con_salt; 
  }

#ifdef WARREN_TEST
  double sigma_solid = 0.05;
  sigma_solid = rho_wall;
  //in_c_plus_wall  = 1.0 - 0.5*sigma_solid;
  rho_p_wall  = 1.0 - 0.5*sigma_solid;
  //in_c_minus_wall = 1.0 + 0.5*sigma_solid;
  rho_m_wall = 1.0 + 0.5*sigma_solid;
  //sigma_fluid = sigma_solid*count_wall/(real)count_fluid;
  double sigma_fluid = sigma_solid * wall_sites / (real) fluid_nodes;
  //in_c_plus_fluid  =  sigma_fluid;
  rho_p_fluid  =  sigma_fluid;
  //in_c_minus_fluid = 0.0;
  rho_m_fluid = 0.0;
  
  printf("\n        ************************************************************\n");
  printf("        *                     WARREN TEST                          *\n");
  printf("        * Init charges distributed as in Warren test.              *\n");
  printf("        * Sigma wall got from tot wallcharge in infile is %f *\n", rho_wall);
  printf("        ************************************************************\n\n");
#endif

  anormf0 = 0.0;
  real factor = params.ch_bjl * PIG;
  /* Distribute charge to WALLS */
  real placed_w_charge = 0.0;
  for( i = 1; i <= lx; i++ ){
    for( j = 1; j <= ly; j++ ){
      index = siz_y * i + j;
      for( k = 0; k < lz; k++ ){
        if( site_type[index][k] == WALL  ){
  	      c_plus [index][k] = rho_p_wall;
	        c_minus[index][k] = rho_m_wall;
	        placed_w_charge += c_plus[index][k] - c_minus[index][k];
        }
      }
    }
  }
#ifndef WARREN_TEST
  printf( "\nPlaced UNIFORMELY total %f charge on WALLS out of the %f input.\n\tWALLS densities (+|-) are (%f|%f)\n", 
      placed_w_charge, wall_charge, rho_p_wall, rho_m_wall );
#endif

  /* Distribute charge to OBJECTS */
  real placed_o_charge = 0.0;
  for( i = 1; i <= lx; i++ ){
    for( j = 1; j <= ly; j++ ){
      index = siz_y * i + j;
      for( k = 0; k < lz; k++ ){
        if( site_type[index][k] >= SOLID  ){
  	      c_plus [index][k] = rho_p_solid;
	        c_minus[index][k] = rho_m_solid;
	        placed_o_charge += c_plus[index][k] - c_minus[index][k];
        }
      }
    }
  }
#ifndef WARREN_TEST
  printf( "Placed UNIFORMELY total %f charge on OBJECTS out of the %f(*%d) input.\nOBJECTS densities (+|-) are (%f|%f)\n", 
      placed_o_charge, params.ch_objects, params.num_obj, rho_p_solid, rho_m_solid );
#endif

  /* Distribute charge to FLUID */
  real placed_f_charge = 0.0;
  int d = params.ch_distype[0];
  /* Cannot allocate in switch: label and statement */
  real ** dh = mem_r_lattice2( ( params.Nx +  2 ) * ( params.Ny +  2 ), params.Nz ); 
  real ** phi_init   = mem_r_lattice2( (params.Nx + 2) * (params.Ny + 2), params.Nz + 2 ); 
  /* Cannot even declare in switch! */
  int go = 1;
  real init_zeta = params.r_pair;
  switch(d){
    case 'u':
      for( i = 1; i <= lx; i++ ){
        for( j = 1; j <= ly; j++ ){
          index = siz_y * i + j;
          for( k = 0; k < lz; k++ ){
            if( site_type[index][k] == FLUID  ){ 
  	          c_plus [index][k] = rho_p_fluid;
	            c_minus[index][k] = rho_m_fluid;
	            placed_f_charge += c_plus[index][k] - c_minus[index][k];
            }
    	      anormf0 += factor * ( c_plus[index][k] + c_minus[index][k] );
          }
        }
      }
#ifndef WARREN_TEST
      printf( "Placed UNIFORMELY total %f charge on FLUID.\n\tFLUID densities (+|-) are (%f|%f) (anormf0 = %e) \n", 
        placed_f_charge, rho_p_fluid, rho_m_fluid, anormf0 );
      printf( "Added %f positive and negative total salt charges\n\tSalt density (+ and - so that total salt charge is zero) is %f on each site.\n\n", con_salt / 2.0 * fluid_nodes, con_salt / 2.0 );
#endif
    break;
    case 'e':
      while( go ){
        dh_potential( dh, init_zeta, params.ch_lambda, objects ) ;
        printf( "\n\t\tEXP-Placing charges exponentially with init_zeta %f\n", init_zeta );
        place_charges_exp( c_plus, c_minus, dh, site_type );
        real eff_zeta = check_zeta_midpoint( phi_init, c_plus, c_minus, site_type );
        double tol_zeta = ( eff_zeta - init_zeta ) / eff_zeta;
        printf( "\t\tEXP-Eff-zeta is %f, tolerance %f\n", eff_zeta, tol_zeta );
        if( fabs(tol_zeta) < 0.05 ){ 
          printf( "\t\tEXP-Tolerance  less than one 5%%.Done\n" ); go = 0; 
        }else{
          real reinit_zeta = init_zeta + ( eff_zeta - init_zeta ) / 10.0;
          printf( "\t\tEXP-Restarting with init-zeta %f\n", reinit_zeta );
          init_zeta = reinit_zeta;
        }
      }
    break;
    case 'M':
      mc_equilibration( c_plus, c_minus, dh, objects, site_type );
      printf( "MC charge placing ended\n" );
      exit( 0 );
    break;    default:
      printf("Init of distribution charge %c not known?! Check input file. Exiting.", d);
      exit( 8 );
  }
  free_2D( (void **) dh );
  free_2D( (void **) phi_init );

  /* Check electroneutrality */
  real tot_ch_p = 0.0;
  real tot_ch_n = 0.0;
  real tot_ch_p_f = 0.0;
  real tot_ch_n_f = 0.0;
  real tot_ch_p_o = 0.0;
  real tot_ch_n_o = 0.0;
  real tot_ch_p_w = 0.0;
  real tot_ch_n_w = 0.0;
  for( i = 1; i <= lx; i++ ){
    for( j = 1; j <= ly; j++ ){
      index = i * siz_y + j;
      for( k = 0; k < lz; k++ ){
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
        if( site_type[index][k] >= SOLID ){
          tot_ch_p_o += c_plus[index][k];
          tot_ch_n_o += c_minus[index][k];
        }
      }
    }
  } 
  real tot_ch = tot_ch_p - tot_ch_n;

  if( fabs( tot_ch ) > 10e-6 ) { 
    printf( "\t\tELECTRONEUTRALITY SCREWED. GOT %.10f TOTAL CHARGE (+%e -%e). CHECK WHAT IS WRONG! BYE\n\n", tot_ch, tot_ch_p, tot_ch_n ); exit ( 0 ); 
  }else{
    printf( "\t\tELECTRONEUTRALITY OK. GOT TOTAL CHARGE OF %e\n", tot_ch );
    printf( "\t\tPUT %e CHARGE on FLUID (+%e %e)\n",  tot_ch_p_f - tot_ch_n_f, tot_ch_p_f, tot_ch_n_f );
    printf( "\t\tPUT %e CHARGE on SOLID (+%e %e)\n",  tot_ch_p_o - tot_ch_n_o, tot_ch_p_o, tot_ch_n_o );
    printf( "\t\tPUT %e CHARGE on WALL (+%e %e)\n",   tot_ch_p_w - tot_ch_n_w, tot_ch_p_w, tot_ch_n_w );
  }
  printf("\n");
  /* I want to see in console log what is going on */
  fflush( stdout );
}

void init_ch_mixture( int ** site_type, double ** c_minus, double ** c_plus, object * objects )
{
  /* ASSUMING THERE ARE NO WALL NODES */
  /* Count solid wall and fluid sites */
  int siz_y = params.Ny + 2;
  int i, j, k, index;
  int tot_sites, obj_sites, wall_sites;
  tot_sites = 0; obj_sites = 0; wall_sites = 0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      index = siz_y * i + j;
      for( k = 0; k < params.Nz; k++ ){
        if( site_type[index][k] >= SOLID ) obj_sites++;
        if( site_type[index][k] == WALL  ) wall_sites++; 
        tot_sites++;
      }
    }
  }
  fluid_nodes = tot_sites - wall_sites - obj_sites;
  printf( "\nLattice of %d sites:\n\t%d wall sites (width %d)\n\t%d object sites\n\t%d fluid sites\n", tot_sites, wall_sites, params.width, obj_sites, fluid_nodes );

  /* Loop over colloids and assign charge to SOLID nodes (depending on sign) */
  real solid_plus  = 0.0;
  real solid_plus_placed  = 0.0;
  real solid_minus = 0.0;
  real solid_minus_placed  = 0.0;
  int n_obj;
  int c_max_num_node = 0;
  for( n_obj = 0; n_obj < params.num_obj; n_obj++ ){
    real q = objects[n_obj].charge;
    solid_plus  = q > 0 ? solid_plus  + q : solid_plus;
    solid_minus = q < 0 ? solid_minus + q : solid_minus;
    real rho_this = q / objects[n_obj].n_points;
    real rho_this_p = rho_this * 0.5;
    real rho_this_m = -rho_this * 0.5;
    int num_nodes_this = 0;
    for( i = 1; i <= params.Nx; i++ ){
      for( j = 1; j <= params.Ny; j++ ){
        index = siz_y * i + j;
        for( k = 0; k < params.Nz; k++ ){
          if( site_type[index][k] == objects[n_obj].self ){
            num_nodes_this++;
  	        c_plus [index][k]  = rho_this_p;
  	        c_minus [index][k] = rho_this_m;
            solid_plus_placed  = rho_this_p > 0.0 ? solid_plus_placed  + rho_this_p : solid_plus_placed;
            solid_minus_placed = rho_this_m > 0.0 ? solid_minus_placed + rho_this_m : solid_minus_placed;
          }
        }
      }
    }
    if( num_nodes_this > c_max_num_node ){
      c_max_num_node = num_nodes_this;
      printf( "\nMax num nodes in a colloid is %d\n", c_max_num_node );
    }
    if( num_nodes_this < c_max_num_node ) printf( "Colloid %d has num_nodes %d\n", objects[n_obj].self, num_nodes_this );
  }
  printf("\nPut %f (out of theor %f) positive SOLID charge and %f (out of theor %f) negative\n", 2.0 * solid_plus_placed, solid_plus, 2.0 * solid_minus_placed, solid_minus );

  /* Work out average positive and negative charge density of fluid nodes due to counterions */
  real rho_cions_fluid_p = 2.0 * solid_plus_placed  / fluid_nodes;
  /* Change sign for electroneutrality */
  rho_cions_fluid_p = -rho_cions_fluid_p;
  real rho_cions_fluid_m = 2.0 * solid_minus_placed / fluid_nodes;
  rho_cions_fluid_m = -rho_cions_fluid_m;

  /* Work out \pm charge density due to the salt */
  real con_salt = 1.0 / ( 4 * PIG * params.ch_bjl * params.ch_lambda * params.ch_lambda );
  if( params.ch_lambda == 0.0 ){  
    con_salt = 0.0;
    printf("\t*****   THIS IS A NO SALT SIMULATION     *****\n");
  }

  real rho_p_fluid = rho_cions_fluid_p + 0.5 * con_salt;
  real rho_m_fluid = rho_cions_fluid_m + 0.5 * con_salt;


  /* Distribute charge to FLUID (uniformely only) */
  real factor = params.ch_bjl * PIG;
  anormf0 = 0.0;
  real placed_f_charge = 0.0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      index = siz_y * i + j;
      for( k = 0; k < params.Nz; k++ ){
        if( site_type[index][k] == FLUID  ){ 
          c_plus [index][k] = rho_p_fluid;
	        c_minus[index][k] = rho_m_fluid;
	        placed_f_charge += c_plus[index][k] - c_minus[index][k];
        }
        anormf0 += factor * ( c_plus[index][k] + c_minus[index][k] );
      }
    }
  }
  printf( "Placed UNIFORMELY total %f charge on FLUID.\n\tFLUID densities (+|-) are (%f|%f) (anormf0 = %e) \n", 
    placed_f_charge, rho_p_fluid, rho_m_fluid, anormf0 );
  printf( "Added %f positive and negative total salt charges\n\tSalt density (+ and - so that total salt charge is zero) is %f on each site.\n\n", con_salt / 2.0 * fluid_nodes, con_salt / 2.0 );

  /* Check electroneutrality */
  real tot_ch_p = 0.0;
  real tot_ch_n = 0.0;
  real tot_ch_p_f = 0.0;
  real tot_ch_n_f = 0.0;
  real tot_ch_p_o = 0.0;
  real tot_ch_n_o = 0.0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        tot_ch_p += c_plus[index][k];
        tot_ch_n += c_minus[index][k];
        if( site_type[index][k] == FLUID ){
          tot_ch_p_f += c_plus[index][k];
          tot_ch_n_f += c_minus[index][k];
        }
        if( site_type[index][k] >= SOLID ){
          tot_ch_p_o += c_plus[index][k];
          tot_ch_n_o += c_minus[index][k];
        }
      }
    }
  } 
  real tot_ch = tot_ch_p - tot_ch_n;

  if( fabs( tot_ch ) > 10e-6 ) { 
    printf( "\t\tELECTRONEUTRALITY SCREWED. GOT %.10f TOTAL CHARGE (+%e -%e). CHECK WHAT IS WRONG! BYE\n", tot_ch, tot_ch_p, tot_ch_n ); 
    printf( "\t\tPUT %e CHARGE on FLUID (+%e %e)\n",  tot_ch_p_f - tot_ch_n_f, tot_ch_p_f, tot_ch_n_f );
    printf( "\t\tPUT %e CHARGE on SOLID (+%e %e)\n\n",  tot_ch_p_o - tot_ch_n_o, tot_ch_p_o, tot_ch_n_o );
    exit( 0 );
  }else{
    printf( "\t\tELECTRONEUTRALITY OK. GOT TOTAL CHARGE OF %e\n", tot_ch );
    printf( "\t\tPUT %e CHARGE on FLUID (+%e %e)\n",  tot_ch_p_f - tot_ch_n_f, tot_ch_p_f, tot_ch_n_f );
    printf( "\t\tPUT %e CHARGE on SOLID (+%e %e)\n",  tot_ch_p_o - tot_ch_n_o, tot_ch_p_o, tot_ch_n_o );
  }
  printf("\n");
  /* I want to see in console log what is going on */
  fflush( stdout );
}

void init_ch_dist_NoCorners( int ** site_type, double **c_minus, double **c_plus, object *objects)
{
  /* CUTTING CORNERS. THIS ASSUMES 1 COLLOID AND CUBIC BOX */
  real objects_charge = params.ch_objects * params.num_obj;
  real wall_charge = params.ch_wall;  
  int siz_y = params.Ny + 2;

  int i, j, k, index;
  /* Transform FLUID nodes in the corners into FLUID nodes */
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      index = siz_y * i + j;
      for( k = 0; k < params.Nz; k++ ){
        real dist = pow( pow((i - objects[0].r.x), 2.0) + pow((j - objects[0].r.y), 2.0) + \
        pow((k - objects[0].r.z),2.0), 0.5 );
        /* -1 so no FLUID node sees the PBC box */
        if( dist > params.Nx / 2.0 - 1.0 )  site_type[index][k] = CORNER;
      }
    }
  }
  /* Count solid wall and fluid sites */
  int tot_sites, obj_sites, wall_sites, corn_sites;
  tot_sites = 0; obj_sites = 0; wall_sites = 0; corn_sites = 0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      index = siz_y * i + j;
      for( k = 0; k < params.Nz; k++ ){
        if( site_type[index][k] >= SOLID ) obj_sites++;
        if( site_type[index][k] == WALL  ) wall_sites++; 
        if( site_type[index][k] == CORNER  ) corn_sites++; 
        tot_sites++;
      }
    }
  }
  fluid_nodes = tot_sites - wall_sites - obj_sites - corn_sites;
  printf("Cut a Sphere of radius %f\n", params.Nx / 2.0 - 1.0 );
  printf( "\nLattice of %d sites:\n\t%d wall sites (width %d)\n\t%d object sites\n\t%d fluid sites\n\t%d corner sites\n", tot_sites, wall_sites, params.width, obj_sites, fluid_nodes, corn_sites );

  /* Work out charge density per wall/solid/fluid site */ 
  real rho_wall, rho_solid;
  real rho_p_solid = 0.0; 
  real rho_m_solid = 0.0;
  real rho_p_wall  = 0.0;
  real rho_m_wall  = 0.0;
  real rho_p_corn  = 0.0;
  real rho_m_corn  = 0.0;
  if( obj_sites ){
    rho_solid = objects_charge / (real) obj_sites;
    rho_p_solid = 0.5 * rho_solid; 
    rho_m_solid = -rho_p_solid;
  }
  if( wall_sites ){
    rho_wall  = wall_charge  / (real) wall_sites;
    rho_p_wall  = 0.5 * rho_wall; 
    rho_m_wall  = -rho_p_wall;
  }

  /* Calculate rho_fluid and change sign to it to guarantee electroneutrality */
  /* Charge on fluid is equal to the charge on obj (if objects) plus charge on wall (if wall) */
  real rho_fluid  = ( params.num_obj ) ? ( objects_charge ) : 0;
  rho_fluid += ( wall_sites ) ? ( wall_charge ) : 0;

  rho_fluid = ( rho_fluid ) / (real) fluid_nodes;
  rho_fluid = -rho_fluid;
  /* Add salt to fluid */
  real con_salt = 1.0 / ( 4 * PIG * params.ch_bjl * params.ch_lambda * params.ch_lambda );

  if( params.ch_lambda == 0.0 ){  
    con_salt = 0.0;
    printf("\t*****   THIS IS A NO SALT SIMULATION     *****\n");
  }

  real rho_p_fluid, rho_m_fluid;
  if( rho_fluid >= 0.0){
    rho_p_fluid = rho_fluid + 0.5 * con_salt; 
    rho_m_fluid = 0.5 * con_salt; 
  }else{
    rho_m_fluid = 0.5 * con_salt - rho_fluid; 
    rho_p_fluid = 0.5 * con_salt; 
  }

  anormf0 = 0.0;
  real factor = params.ch_bjl * PIG;
  /* Distribute charge to WALLS */
  real placed_w_charge = 0.0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      index = siz_y * i + j;
      for( k = 0; k < params.Nz; k++ ){
        if( site_type[index][k] == WALL  ){
  	      c_plus [index][k] = rho_p_wall;
	        c_minus[index][k] = rho_m_wall;
	        placed_w_charge += c_plus[index][k] - c_minus[index][k];
        }
      }
    }
  }
  /* Distribute charge to OBJECTS */
  real placed_o_charge = 0.0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      index = siz_y * i + j;
      for( k = 0; k < params.Nz; k++ ){
        if( site_type[index][k] >= SOLID  ){
  	      c_plus [index][k] = rho_p_solid;
	        c_minus[index][k] = rho_m_solid;
	        placed_o_charge += c_plus[index][k] - c_minus[index][k];
        }
      }
    }
  }
  /* Distribute charge to CORNER */
  real placed_c_charge = 0.0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      index = siz_y * i + j;
      for( k = 0; k < params.Nz; k++ ){
        if( site_type[index][k] == CORNER  ){
  	      c_plus [index][k] = rho_p_corn;
	        c_minus[index][k] = rho_m_corn;
	        placed_c_charge += c_plus[index][k] - c_minus[index][k];
        }
      }
    }
  }
  /* Distribute charge to FLUID */
  real placed_f_charge = 0.0;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      index = siz_y * i + j;
      for( k = 0; k < params.Nz; k++ ){
        if( site_type[index][k] == FLUID  ){ 
          c_plus [index][k] = rho_p_fluid;
	        c_minus[index][k] = rho_m_fluid;
	        placed_f_charge += c_plus[index][k] - c_minus[index][k];
        }
        anormf0 += factor * ( c_plus[index][k] + c_minus[index][k] );
      }
    }
  }
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
      index = i * siz_y + j;
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
        if( site_type[index][k] >= SOLID ){
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
    printf( "\t\tELECTRONEUTRALITY SCREWED. GOT %.10f TOTAL CHARGE (+%e -%e). CHECK WHAT IS WRONG! BYE\n\n", tot_ch, tot_ch_p, tot_ch_n ); exit ( 0 ); 
  }else{
    printf( "\t\tELECTRONEUTRALITY OK. GOT TOTAL CHARGE OF %e\n", tot_ch );
    printf( "\t\tPUT %e CHARGE on FLUID (+%e %e)\n",  tot_ch_p_f - tot_ch_n_f, tot_ch_p_f, tot_ch_n_f );
    printf( "\t\tPUT %e CHARGE on SOLID (+%e %e)\n",  tot_ch_p_o - tot_ch_n_o, tot_ch_p_o, tot_ch_n_o );
    printf( "\t\tPUT %e CHARGE on WALL (+%e %e)\n",   tot_ch_p_w - tot_ch_n_w, tot_ch_p_w, tot_ch_n_w );
    printf( "\t\tPUT %e CHARGE on CORNERS (+%e %e)\n",   tot_ch_p_c - tot_ch_n_c, tot_ch_p_c, tot_ch_n_c );
  }
  printf("\n");
  /* I want to see in console log what is going on */
  fflush( stdout );
}


inline real phi_edl_sphere( int i, int j, int k, real charge_sph, object obj )
/* Calculation of EDL potential field assuming PBC */
{
  //real dist = fabs(i - obj.r.x);
  //real rx_val = ( dist <= max_x / 2.0 )? dist : max_x - dist;
  real rx_val = fabs( i - obj.r.x );
  rx_val = min( rx_val, fabs(params.Nx - rx_val) );
  real ry_val = fabs( j - obj.r.y );
  ry_val = min( ry_val, fabs(params.Ny - ry_val) );
  real rz_val = fabs( k - obj.r.z );
  rz_val = min( rz_val, fabs(params.Nz - rz_val) );
  real r = sqrt( rx_val * rx_val + ry_val * ry_val + rz_val * rz_val );
  //real v_phi = charge_sph * exp ( params.r_eff / params.ch_lambda ) / ( 1 + params.r_eff / params.ch_lambda ) * params.ch_bjl * exp ( -r / params.ch_lambda ) / r; 
  real v_phi = 100.0 * charge_sph * params.r_eff * exp ( -r / params.ch_lambda ) / ( r + params.r_eff ) ; 
  return v_phi;
}

/*************************************************************/
float gasdev(long *idum)
/*************************************************************/
{
	float ran1(long *idum);
	static int iset=0;
	static float gset;
	float fac,rsq,v1,v2;

	if  (iset == 0) {
		do {
			v1=2.0*ran1(idum)-1.0;
			v2=2.0*ran1(idum)-1.0;
			rsq=v1*v1+v2*v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac=sqrt(-2.0*log(rsq)/rsq);
		gset=v1*fac;
		iset=1;
		return v2*fac;
	} else {
		iset=0;
		return gset;
	}
}
/* (C) Copr. 1986-92 Numerical Recipes Software "0j. */
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

float ran1(long *idum)
{
	int j;
	long k;
	static long iy=0;
	static long iv[NTAB];
	float temp;

	if (*idum <= 0 || !iy) {
		if (-(*idum) < 1) *idum=1;
		else *idum = -(*idum);
		for (j=NTAB+7;j>=0;j--) {
			k=(*idum)/IQ;
			*idum=IA*(*idum-k*IQ)-IR*k;
			if (*idum < 0) *idum += IM;
			if (j < NTAB) iv[j] = *idum;
		}
		iy=iv[0];
	}
	k=(*idum)/IQ;
	*idum=IA*(*idum-k*IQ)-IR*k;
	if (*idum < 0) *idum += IM;
	j=iy/NDIV;
	iy=iv[j];
	iv[j] = *idum;
	if ((temp=AM*iy) > RNMX) return RNMX;
	else return temp;
}

#undef IA
#undef IM
#undef AM
#undef IQ
#undef IR
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX
/* (C) Copr. 1986-92 Numerical Recipes Software "0j. */

/*************************************************************/
int    nint2( float x) 
/*************************************************************/
{
  if( x >= 0.0 )
    {
      return (int)((x)+(0.5));
    } 
  else
    {
      return (int)((x)-(0.5));
    }
}

