/*****************************************************************************
 *
 *  propagation_gpu.h
 *
 *  Alan Gray
 *
 *****************************************************************************/

#ifndef _PROPAGATION_GPU_H

#ifdef CSRC
#define CFUNC 
#else
#define CFUNC extern "C"
#endif
#define _PROPAGATION_GPU_H

/* expose main routine in this module to outside routines */
CFUNC void propagation_gpu();

#endif
