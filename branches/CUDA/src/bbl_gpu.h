/*****************************************************************************
 * 
 * bbl_gpu.h
 * 
 * Alan Gray
 *
 *****************************************************************************/

#ifndef BBL_GPU_H
#define BBL_GPU_H

#ifdef CSRC
#define CFUNC 
#else
#define CFUNC extern "C"
#endif

/* expose routines in this module to outside routines */   
CFUNC void bounce_back_gpu(int *findexall, int *linktype,               
                            double *dfall, double *dgall1, double *dgall2,  
                                double *dmall, int nlink, int pass);       
CFUNC void bbl_init_temp_link_arrays_gpu(int nlink);                    
CFUNC void bbl_finalise_temp_link_arrays_gpu();                    
CFUNC void bbl_enlarge_temp_link_arrays_gpu(int nlink);             

#endif
