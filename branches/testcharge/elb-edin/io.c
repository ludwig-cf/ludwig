#include <stdio.h>
#include <string.h>
#include "io.h"
#include "init.h"
#include "simpar.h"

extern sim_params_t params;

extern int x_dir[18];
extern int y_dir[18];
extern int z_dir[18];
extern int fluid_nodes;

FILE * init_file( char prx[32], char * sfx )
{
  FILE * file;
  char buffer[64];
  sprintf( buffer, prx );
  strcat( buffer, sfx );
  file = fopen( buffer, "w" );
  return file;
}

void vtk_dump_lattice_scalar( real ** s_field, char * fname )
{
  /* GG TODO WARNING note that for the weird data structure of lattice X and Z axis are swapped */
  FILE * file = fopen( fname, "w" );
  int siz_y = params.Ny + 2;
fprintf(file,"# vtk DataFile Version 2.0\n\
%s\n\
ASCII\n\
DATASET STRUCTURED_POintS\n\
DIMENSIONS %d %d %d\n\
ORIGIN 0 0 0\n\
SPACING 1 1 1\n\
POint_DATA %d\n\
SCALARS Scalars0 float 1\n\
LOOKUP_TABLE default\n", 
fname,
params.Nx, params.Ny, params.Nz, 
params.Nx * params.Ny * params.Nz
);
  int i, j, k;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        fprintf(file,"%f\n",s_field[index][k]);
      }
    }
  }
  fclose( file );
}

void write_lattice( real *** velcs_ptr, int ** site_type, real ** c_plus, real ** c_minus, 
    real ** phi, real ** phi_p_x, real ** phi_p_y, real ** phi_p_z, int n_step, char * string )
{
  FILE * f_lattice;
  int siz_y = params.Ny + 2;
  char buffer[128];
  if( n_step != 888 ) sprintf( buffer, "oelb_lattice_%s_%d.elb", string, n_step );
  if( n_step == 888 ) sprintf( buffer, "oelb_lattice_%s_GK.elb", string );
  f_lattice = fopen(buffer,"w"); 

  int i, j, k;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = i * siz_y + j;
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
        /* Commented to decrease .dat output files 
        char to_print[256] = {};
        for( velc_dir = 0; velc_dir < 18; velc_dir++){
          char to_add[16] = {};
          sprintf( to_add, " %f", velcs_ptr[index][velc_dir][k] );
          strcat( to_print, to_add );
        }
        */
        int what   = site_type[index][k];
        /* Note that charges and potential fields have different indexing system as they do not have haloes */
        //int inx = i * params.Ny + j;
        real c_p  = c_plus[index][k];
        real c_m  = c_minus[index][k];
        real pot  = phi[index][k+1];
        real pot_p_x  = phi_p_x[index][k+1];
        real pot_p_y  = phi_p_y[index][k+1];
        real pot_p_z  = phi_p_z[index][k+1];
        /* Commented to decrease .dat output files
        fprintf( f_lattice, "%d %d %d %f %e %e %e   %d   %e %e %e %e %e %e %e @ %s\n", i, j, k, tot_mass, tot_px, tot_py, tot_pz, what, pot, pot_p_x, pot_p_y, pot_p_z, c_p, c_m, c_p-c_m, to_print ); 
        */
        fprintf( f_lattice, "%d %d %d %e %e %e   %d   %e %e %e %e %e %e %e\n", i, j, k, tot_px, tot_py, tot_pz, what, pot, pot_p_x, pot_p_y, pot_p_z, c_p, c_m, c_p-c_m );
      }
    }
  }
  fclose ( f_lattice );
}

void dump_bin_lattice( real *** velcs_ptr, int ** site_type, real ** c_plus, real ** c_minus, 
    real ** phi, real ** phi_p_x, real ** phi_p_y, real ** phi_p_z, int n_step, char * string )
{
  FILE * f_bin;
  char binbuf[128];
  sprintf( binbuf, "oelb_lattice_%s_%d.bin", string, n_step );
  f_bin = fopen(binbuf,"w"); 
  fwrite( velcs_ptr[0][0], sizeof( real ), (params.Nx + 2) * (params.Ny + 2) * 18 * params.Nz, f_bin );
  fwrite( phi[0], sizeof( real ), (params.Nx + 2) * (params.Ny + 2) * (params.Nz + 2), f_bin );
  fwrite( phi_p_x[0], sizeof( real ), (params.Nx + 2) * (params.Ny + 2) * (params.Nz + 2), f_bin );
  fwrite( phi_p_y[0], sizeof( real ), (params.Nx + 2) * (params.Ny + 2) * (params.Nz + 2), f_bin );
  fwrite( phi_p_z[0], sizeof( real ), (params.Nx + 2) * (params.Ny + 2) * (params.Nz + 2), f_bin );
  fwrite( c_plus[0], sizeof( real ), (params.Nx + 2) * (params.Ny + 2) * params.Nz, f_bin );
  fwrite( c_minus[0], sizeof( real ), (params.Nx + 2) * (params.Ny + 2) * params.Nz, f_bin );
  fwrite( site_type[0], sizeof( int ), (params.Nx + 2) * (params.Ny + 2) * params.Nz, f_bin );
  fclose ( f_bin );
}

void load_bin_lattice( char * binfile, real *** velcs_ptr, int ** site_type, real ** c_plus, 
    real ** c_minus, real ** phi, real ** phi_p_x, real ** phi_p_y, real ** phi_p_z )
{
  FILE * f_bin;
  f_bin = fopen( binfile, "r" ); 
  if( !f_bin ) printf( "Unable to load file %s\n", binfile );
  fread( velcs_ptr[0][0], sizeof( real ), (params.Nx + 2) * (params.Ny + 2) * 18 * params.Nz, f_bin );
  fread( phi[0], sizeof( real ), (params.Nx + 2) * (params.Ny + 2) * (params.Nz + 2), f_bin );
  fread( phi_p_x[0], sizeof( real ), (params.Nx + 2) * (params.Ny + 2) * (params.Nz + 2), f_bin );
  fread( phi_p_y[0], sizeof( real ), (params.Nx + 2) * (params.Ny + 2) * (params.Nz + 2), f_bin );
  fread( phi_p_z[0], sizeof( real ), (params.Nx + 2) * (params.Ny + 2) * (params.Nz + 2), f_bin );
  fread( c_plus[0], sizeof( real ), (params.Nx + 2) * (params.Ny + 2) * params.Nz, f_bin );
  fread( c_minus[0], sizeof( real ), (params.Nx + 2) * (params.Ny + 2) * params.Nz, f_bin );
  fread( site_type[0], sizeof( int ), (params.Nx + 2) * (params.Ny + 2) * params.Nz, f_bin );
  fclose ( f_bin );
}


void write_colloids( object * objects, real * J, int n_step, char * string )
{
  FILE * f_colloid;
  char buffer[128];
  sprintf( buffer, "oelb_colloids_%s.elb", string );
  f_colloid = fopen(buffer,"a"); 

  int n_obj;
  for( n_obj = 0; n_obj < params.num_obj; n_obj++ ){
    object col = objects[n_obj];
    if( !params.GreenKubo ){
      fprintf( f_colloid, "%d %d %d %d %d %f %f %f %e %e %e %f %f %f %f %f %f %f %f %f\n", n_step, n_obj, col.i.x, col.i.y, col.i.z, col.r.x, col.r.y, col.r.z, col.u.x, col.u.y, col.u.z, col.f.x, col.f.y, col.f.z, col.w.x, col.w.y, col.w.z, col.t.x, col.t.y, col.t.z );
    }else{
      fprintf( f_colloid, "%d %d %d %d %d %f %f %f %e %e %e %e %e %e %f %f %f %f %f %f %f %f %f\n", n_step, n_obj, col.i.x, col.i.y, col.i.z, col.r.x, col.r.y, col.r.z, col.u.x, col.u.y, col.u.z, J[0], J[1], J[2], col.f.x, col.f.y, col.f.z, col.w.x, col.w.y, col.w.z, col.t.x, col.t.y, col.t.z );
    }
  }
  fclose( f_colloid );
}

void vtk_write_lattice( real *** velcs_ptr, int ** site_type, real ** c_plus, real ** c_minus, real ** phi, int n_step, char * string )
{
  /* GG TODO WARNING note that for the weird data structure of lattice X and Z axis are swapped */
  FILE * f_vtk_lattice;
  int siz_y = params.Ny + 2;
  char buffer[128];
  sprintf( buffer, "oelb_lattice_%s_%d.vtk", string, n_step );
  f_vtk_lattice = fopen(buffer,"w"); 
fprintf(f_vtk_lattice,"# vtk DataFile Version 2.0\n\
%s\n\
ASCII\n\
DATASET STRUCTURED_POintS\n\
DIMENSIONS %d %d %d\n\
ORIGIN 0 0 0\n\
SPACING 1 1 1\n\
POint_DATA %d\n\
SCALARS TotCharge float 1\n\
LOOKUP_TABLE default\n", 
buffer,
params.Nx, params.Ny, params.Nz, 
params.Nx * params.Ny * params.Nz
);
  int i, j, k;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        fprintf( f_vtk_lattice, "%f\n", c_plus[index][k] - c_minus[index][k] );
      }
    }
  }

  fprintf( f_vtk_lattice, "VECTORS Velocity float\n" );
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        float tot_px, tot_py, tot_pz;
        int velc_dir;
        tot_px = tot_py = tot_pz = 0.0;
        for( velc_dir = 0; velc_dir < 18; velc_dir++){
          tot_px   += velcs_ptr[index][velc_dir][k] * x_dir[velc_dir];
          tot_py   += velcs_ptr[index][velc_dir][k] * y_dir[velc_dir];
          tot_pz   += velcs_ptr[index][velc_dir][k] * z_dir[velc_dir];
        }
        fprintf( f_vtk_lattice, "%f %f %f\n", tot_px, tot_py, tot_pz );
      }
    }
  }

  fclose( f_vtk_lattice );
}


void vtk_dump_lattice_charge( real ** plus, real ** minus, char * fname )
{
  int siz_y = params.Ny + 2;
  /* GG TODO WARNING note that for the weird data structure of lattice X and Z axis are swapped */
  FILE * file = fopen( fname, "w" );
fprintf(file,"# vtk DataFile Version 2.0\n\
%s\n\
ASCII\n\
DATASET STRUCTURED_POintS\n\
DIMENSIONS %d %d %d\n\
ORIGIN 0 0 0\n\
SPACING 1 1 1\n\
POint_DATA %d\n\
SCALARS Scalars0 float 1\n\
LOOKUP_TABLE default\n", 
fname,
params.Nx, params.Ny, params.Nz, 
params.Nx * params.Ny * params.Nz
);
  int i, j, k;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      int index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        fprintf( file, "%f\n", plus[index][k] - minus[index][k] );
      }
    }
  }
  fclose( file );
}

void write_fluxes( real *** velcs_ptr, int ** site_type, real ** c_plus, real ** c_minus, int n_step )
{
  extern int y_dir[18], z_dir[18];
  real flux_rhopy = 0.0; 
  real flux_rhomy = 0.0; 
  real flux_rhopz = 0.0; 
  real flux_rhomz = 0.0;
  real max_x = params.Nx;
  real max_y = params.Ny;
  real max_z = params.Nz;
  int siz_y = params.Ny + 2;
  int x, y, z;
  int velc_dir;
  for( x = 1; x <= max_x; x++ ){
    for( y = 1; y <= max_y; y++ ){ 
      int index = x * siz_y + y;
      for( z=0; z < max_z; z++ ){
        if( site_type[index][z]==FLUID ){
          for( velc_dir = 0; velc_dir < 18; velc_dir++ ){
            flux_rhopy +=  *( velcs_ptr[index][velc_dir] + z ) * y_dir[velc_dir] * c_plus[index][z];
            flux_rhomy +=  *( velcs_ptr[index][velc_dir] + z ) * y_dir[velc_dir] * c_minus[index][z];
            flux_rhopz +=  *( velcs_ptr[index][velc_dir] + z ) * z_dir[velc_dir] * c_plus[index][z];
            flux_rhomz +=  *( velcs_ptr[index][velc_dir] + z ) * z_dir[velc_dir] * c_minus[index][z];
          }
        }
      }
    }
  }
  /*
  fprintf( f_fluxos, "%d %e %e %e %e %e %e %e %e\n", n_step,
	flux_rhopy / ( params.rho0*fluid_nodes ), flux_rhomy / ( params.rho0*fluid_nodes ),
	(flux_rhopy - flux_rhomy) / (params.rho0 * fluid_nodes), (flux_rhopy + flux_rhomy) / (params.rho0 * fluid_nodes),
	flux_rhopz / (params.rho0 * fluid_nodes), flux_rhomz / (params.rho0 * fluid_nodes),
	(flux_rhopz - flux_rhomz) / (params.rho0 * fluid_nodes), (flux_rhopz + flux_rhomz) / (params.rho0 * fluid_nodes) );
  */
  printf("Step %d: Fluxes %e %e %e %e %e %e %e %e\n",n_step,
	flux_rhopy/(params.rho0*fluid_nodes), flux_rhomy/(params.rho0*fluid_nodes),
	(flux_rhopy-flux_rhomy)/(params.rho0*fluid_nodes), (flux_rhopy+flux_rhomy)/(params.rho0*fluid_nodes),
	flux_rhopz/(params.rho0*fluid_nodes), flux_rhomz/(params.rho0*fluid_nodes),
	(flux_rhopz-flux_rhomz)/(params.rho0*fluid_nodes), (flux_rhopz+flux_rhomz)/(params.rho0*fluid_nodes));
}

void add_charge_fluid( real ** c_plus, real ** c_minus, int ** site_type, real * missing_charge, int fluid_nodes )
{
  double to_add_p = missing_charge[0] / (real) fluid_nodes;
  double to_add_m = missing_charge[1] / (real) fluid_nodes;
  int i, j, k, index;
  int siz_y = params.Ny + 2;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        if( site_type[index][k] == FLUID ){
          c_plus[index][k]  += to_add_p;
          c_minus[index][k] += to_add_m;
        }
      }
    }
  }
}

void check_charge_fluid( real ** c_plus, real ** c_minus, int ** site_type, real * tot_charge )
{
  double tot_ch_p, tot_ch_m;
  tot_ch_p = tot_ch_m = 0.0;
  int i, j, k, index;
  int siz_y = params.Ny + 2;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        if( site_type[index][k] == FLUID ){
          tot_ch_p += c_plus[index][k];
          tot_ch_m += c_minus[index][k];
        }
      }
    }
  }
  tot_charge[0] = tot_ch_p;
  tot_charge[1] = tot_ch_m;
}

void check_conservations( real *** velcs_ptr, int ** site_type, real ** c_plus, real ** c_minus, double * res, object * objects )
{
  double tot_ch_p, tot_ch_m, tot_mass, tot_px, tot_py, tot_pz;
  tot_ch_p = tot_ch_m = tot_mass = tot_px = tot_py = tot_pz = 0.0;
  int i, j, k, index, velc_dir;
  int siz_y = params.Ny + 2;
  for( i = 1; i <= params.Nx; i++ ){
    for( j = 1; j <= params.Ny; j++ ){
      index = i * siz_y + j;
      for( k = 0; k < params.Nz; k++ ){
        //if( site_type[index][k] == FLUID ){
          tot_ch_p += c_plus[index][k];
          tot_ch_m += c_minus[index][k];
          for( velc_dir = 0; velc_dir < 18; velc_dir++){
            tot_mass += velcs_ptr[index][velc_dir][k];
            tot_px += velcs_ptr[index][velc_dir][k] * x_dir[velc_dir];
            tot_py += velcs_ptr[index][velc_dir][k] * y_dir[velc_dir];
            tot_pz += velcs_ptr[index][velc_dir][k] * z_dir[velc_dir];
          }
        //}
      }
    }
  }

  int n_obj;
  real tot_px_coll, tot_py_coll, tot_pz_coll;
  tot_px_coll = tot_py_coll = tot_pz_coll = 0.0;
  for( n_obj = 0; n_obj < params.num_obj; n_obj++ ){
    tot_px_coll +=  objects[n_obj].u.x * objects[n_obj].mass;
    tot_py_coll +=  objects[n_obj].u.y * objects[n_obj].mass;
    tot_pz_coll +=  objects[n_obj].u.z * objects[n_obj].mass;
  }

  res[0] = tot_mass;
  res[1] = tot_ch_p - tot_ch_m;
  res[2] = tot_px;
  res[3] = tot_py;
  res[4] = tot_pz;
  res[5] = tot_px_coll;
  res[6] = tot_py_coll;
  res[7] = tot_pz_coll;
  res[8] = tot_ch_p;
  res[9] = tot_ch_m;
}

