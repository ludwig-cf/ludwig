#ifndef _IO_H
#define _IO_H
#include "lb_types.h"
#include "lb.h"

FILE * init_file( char prx[32], char * sfx );
void write_fluxes( real *** velcs_ptr, int ** site_type, real ** c_plus, real ** c_minus, int n_step );
void write_lattice( real *** velcs_ptr, int ** site_type, real ** c_plus, real ** c_minus, 
    real ** phi, real ** phi_p_x, real ** phi_p_y, real ** phi_p_z, int n_step, char * string );
void dump_bin_lattice( real *** velcs_ptr, int ** site_type, real ** c_plus, real ** c_minus, 
    real ** phi, real ** phi_p_x, real ** phi_p_y, real ** phi_p_z, int n_step, char * string );
void load_bin_lattice( char * binfile, real *** velcs_ptr, int ** site_type, real ** c_plus, 
    real ** c_minus, real ** phi, real ** phi_p_x, real ** phi_p_y, real ** phi_p_z );
void write_colloids( object * objects, real * J, int n_step, char * string );
void vtk_write_lattice( real *** velcs_ptr, int ** site_type, real ** c_plus, real ** c_minus, real ** phi, int n_step, char * string );
void vtk_dump_lattice_charge( real ** plus, real ** minus, char * fname );
void vtk_dump_lattice_scalar( real ** s_field, char * fname );
void vtk_dump_lattice_charge( real ** plus, real ** minus, char * fname );
void add_charge_fluid( real ** c_plus, real ** c_minus, int ** site_type, real * missing_charge, int fluid_nodes );
void check_charge_fluid( real ** c_plus, real ** c_minus, int ** site_type, real * tot_charge );
void check_conservations( real *** velcs_ptr, int ** site_type, real ** c_plus, real ** c_minus, double * res, object * objects );

#endif /* IO */
