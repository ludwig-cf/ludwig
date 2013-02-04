/*****************************************************************************
 *
 *  phi_force_gpu.h
 *
 *  Alan Gray
 *
 *****************************************************************************/

#ifndef _PHI_FORCE_GPU_H
#define _PHI_FORCE_GPU_H

#ifdef CSRC
#define CFUNC 
#else
#define CFUNC extern "C"
#endif

/* expose main routine in this module to outside routines */
CFUNC void phi_force_calculation_gpu(void);
CFUNC void phi_force_colloid_gpu(void);
CFUNC void blue_phase_be_update_gpu(void);
CFUNC void advection_upwind_gpu(void);
CFUNC void advection_bcs_no_normal_flux_gpu(void);


#endif
