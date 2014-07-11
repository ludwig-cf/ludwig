/*****************************************************************************
 * 
 * utilities_gpu.h
 * 
 * Alan Gray
 *
 *****************************************************************************/

#ifndef UTILITIES_GPU_H
#define UTILITIES_GPU_H

#include "colloid.h"

#ifdef __cplusplus
extern "C" {
#endif

void initialise_gpu();
void put_site_map_on_gpu();
void put_colloid_map_on_gpu();
void put_colloid_properties_on_gpu();
void zero_colloid_force_on_gpu();
void get_fluxes_from_gpu();
void put_fluxes_on_gpu();
void finalise_gpu();
void checkCUDAError(const char *msg);
int get_linear_index(int ii,int jj,int kk,int N[3]);

/* KEVIN */

int colloids_to_gpu(void);
int colloids_gpu_force_reduction(void);

typedef struct coll_array_s coll_array_t;

struct coll_array_s {
  int nc;                        /* Current number of local colloids */
  int ncmax;                     /* High water mark */
  int nblock[3];                 /* No. thread blocks */
  int * mapd;                    /* Site -> colloid index map */
  colloid_state_t * s;           /* Array of states */
  double * fblockd;              /* Reduction data for force on colloids. */
};


#ifdef __cplusplus
}
#endif

#endif
