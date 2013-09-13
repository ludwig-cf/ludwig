/*****************************************************************************
 *
 *  collision_gpu.h
 *
 *  Alan Gray
 * 
 *****************************************************************************/

#ifndef COLLISION_GPU_H
#define COLLISION_GPU_H

#ifdef CSRC
#define CFUNC 
#else
#define CFUNC extern "C"
#endif

/* expose main routine in this module to outside routines */
CFUNC void collide_gpu(int async);
CFUNC void collide_bulk_gpu();
CFUNC void collide_edges_gpu();
CFUNC void collide_wait_gpu(int async);
CFUNC void launch_bulk_calc_gpu();



#endif
