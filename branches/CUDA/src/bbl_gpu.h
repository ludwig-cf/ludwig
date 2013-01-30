/*****************************************************************************
 * 
 * bbl_gpu.h
 * 
 * Alan Gray
 *
 *****************************************************************************/

#ifndef BBL_GPU_H
#define BBL_GPU_H

#include "common_gpu.h"
#include "model.h"

/* expose routines in this module to outside routines */
extern "C" void bounce_back_gpu(int *findexall, int *linktype,
				double *dfall, double *dgall1, double *dgall2,
				double *dmall, int nlink, int pass);
extern "C" void bbl_init_temp_link_arrays_gpu(int nlink);
extern "C" void bbl_finalise_temp_link_arrays_gpu();
extern "C" void bbl_enlarge_temp_link_arrays_gpu(int nlink);
#endif
