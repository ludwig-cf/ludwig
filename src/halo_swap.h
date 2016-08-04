/*****************************************************************************
 *
 *  halo_swap.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2016 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_HALO_SWAP_H
#define LUDWIG_HALO_SWAP_H


#include "kernel.h"

typedef struct halo_swap_s halo_swap_t;

/* Could be void * data with MPI_Datatype if more general case required */

typedef void (*f_pack_t)(halo_swap_t * halo, int id, double * data);
typedef void (*f_unpack_t)(halo_swap_t * halo, int id, double * data);

__host__ int halo_swap_create_r1(int nhcomm, int naddr, int na, halo_swap_t ** phalo);
__host__ int halo_swap_create_r2(int nhcomm, int naddr, int na, int nb, halo_swap_t ** phalo);
__host__ int halo_swap_free(halo_swap_t * halo);
__host__ int halo_swap_commit(halo_swap_t * halo);
__host__ int halo_swap_handlers_set(halo_swap_t * halo, f_pack_t pack, f_unpack_t unpack);
__host__ int halo_swap_host_rank1(halo_swap_t * halo, void * mbuf,
				  MPI_Datatype mpidata);
__host__ int halo_swap_packed(halo_swap_t * halo, double * data);

__global__ void halo_swap_pack_rank1(halo_swap_t * halo, int id, double * data);
__global__ void halo_swap_unpack_rank1(halo_swap_t * halo, int id, double * data);

#endif
