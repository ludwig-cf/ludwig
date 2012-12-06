/*****************************************************************************
 *
 *  interface_gpu.h
 *
 *  Forward declarations of CUDA-wrapping C routines  
 * 
 *  Alan Gray
 *
 *****************************************************************************/

#ifndef _INTERFACE_GPU_H
#define _INTERFACE_GPU_H

void initialise_gpu(void);
void put_site_map_on_gpu(void);
void put_f_on_gpu(void);
void put_f_partial_on_gpu(int *mask, int include_neighbours);
void put_force_on_gpu(void);
void put_phi_on_gpu(void);
void put_grad_phi_on_gpu(void);
void put_delsq_phi_on_gpu(void);
void put_velocity_on_gpu(void);
void get_f_from_gpu(void);
void get_g_partial_from_gpu(int *mask, int include_neighbours);
void get_force_from_gpu(void);
void zero_force_on_gpu(void);
void get_velocity_from_gpu(void);
void get_phi_from_gpu(void);
void finalise_gpu(void);
void expand_phi_on_gpu(void);
void collide_gpu(void);
void propagation_gpu(void);
void phi_compute_phi_site_gpu(void);
void distribution_halo_gpu(void);
void phi_halo_gpu(void);
void phi_gradients_compute_gpu(void);
void bounce_back_gpu(int *findexall, int *linktype, double *dfall,
		     double *dmall, double *dgall1,double *dgall2,
		     int nlink, int pass);
void bbl_init_temp_link_arrays_gpu(int nlink);
void bbl_finalise_temp_link_arrays_gpu();
void bbl_enlarge_temp_link_arrays_gpu(int nlink);

#endif
