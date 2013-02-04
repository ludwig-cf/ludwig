/*****************************************************************************
 *
 *  phi_lb_coupler_gpu.h
 *
 *  Alan Gray
 *
 *****************************************************************************/

#ifndef _PHI_LB_COUPLER_GPU_H
#define _PHI_LB_COUPLER_GPU_H


#ifdef CSRC
#define CFUNC 
#else
#define CFUNC extern "C"
#endif

/* expose main routine in this module to outside routines */
CFUNC void phi_compute_phi_site_gpu();

#endif
