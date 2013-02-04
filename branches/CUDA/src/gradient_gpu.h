/*****************************************************************************
 *
 *  gradient_gpu.h
 *
 *  Alan Gray
 *
 *****************************************************************************/

#ifndef _GRADIENT_GPU_H
#define _GRADIENT_GPU_H

#ifdef CSRC
#define CFUNC 
#else
#define CFUNC extern "C"
#endif

/* expose routines in this module to outside routines */
CFUNC void phi_gradients_compute_gpu(void);
CFUNC void set_gradient_option_gpu(char option);

#endif
