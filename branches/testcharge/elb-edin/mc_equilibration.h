#ifndef _MC_EQUILIBRATION_H
#define _MC_EQUILIBRATION_H

void randomize_charges( double ** c_plus, double ** c_minus, int ** site_type );
void dh_potential( real ** dh, double zeta, double lambda_d, object * obj );
void place_charges_exp( real ** c_plus, real ** c_minus, real ** dh, int ** site_type );
//void mc_moves( real * c1_plus, real * c1_minus, int ** nns, int * site1_type );
void check_electroneutrality( double ** c_plus, double ** c_minus, int ** site_type );
int create_c1_data( real ** c_plus, real ** c_minus, int ** site_type, real * c1_plus, real * c1_minus, int * site1_type, int * dual );
int create_dual( int * dual );
void create_distances( real cubic_cutoff, int * dual, real ** distance );
void create_nns( int mc_sites, int * dual, int ** nns );
void do_nn( int a, int dual, int * nns );
void mc_equilibration( double ** c_plus, double ** c_minus, real ** dh, object  * objects, int ** site_type );
void mc_moves( real ** c_plus, real ** c_minus, int ** nns, int ** site_type, int * dual );
real calc_delta_energy( double ** c_plus, double ** c_minus, int as, int bs, int cs, int an, int bn, int cn, real delta_ch, real try_delta_ch );
real check_zeta_midpoint( real ** phi, real ** c_plus, real ** c_minus, int ** site_type );

#endif /* MC_EQUILIBRATION */
