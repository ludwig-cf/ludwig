/*****************************************************************************
 * 
 * comms_gpu.h
 * 
 * Alan Gray
 *
 *****************************************************************************/

#ifndef COMMS_GPU_H
#define COMMS_GPU_H

#ifdef CSRC
#define CFUNC 
#else
#define CFUNC extern "C"
#endif

/* expose routines in this module to outside routines */
CFUNC void init_comms_gpu();
CFUNC void finalise_comms_gpu();
//CFUNC void getXYZstreamptr(void* Xptr,void* Yptr,void* Zptr);

#ifndef CSRC
CFUNC cudaStream_t getXstream();
CFUNC cudaStream_t getYstream();
CFUNC cudaStream_t getZstream();
#endif

//CFUNC void get_XYZstreams(cudaStream_t *streamXptr,cudaStream_t *streamYptr,cudaStream_t *streamZptr);
CFUNC void halo_gpu(int nfields1, int nfields2, int packfield1, double * data_d);
CFUNC void put_field_partial_on_gpu(int nfields1, int nfields2, int include_neighbours,double *data_d, void (* access_function)(const int, double *));

CFUNC void get_field_partial_from_gpu(int nfields1, int nfields2, int include_neighbours,double *data_d, void (* access_function)(const int, double *));

#endif

