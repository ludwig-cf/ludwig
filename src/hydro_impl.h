/*****************************************************************************
 *
 *  hydro_impl.h
 *
 *  Static inline functions.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_HYDRO_IMPL_H
#define LUDWIG_HYDRO_IMPL_H

#include "hydro.h"

/*****************************************************************************
 *
 *  hydro_u_set
 *
 *****************************************************************************/

__host__ __device__ static inline void hydro_u_set(hydro_t * hydro,
						   int index,
						   const double u[3]) {
  assert(hydro);
  hydro->u->data[addr_rank1(hydro->nsite, 3, index, X)] = u[X];
  hydro->u->data[addr_rank1(hydro->nsite, 3, index, Y)] = u[Y];
  hydro->u->data[addr_rank1(hydro->nsite, 3, index, Z)] = u[Z];

  return;
}

/*****************************************************************************
 *
 *  hydro_u
 *
 *****************************************************************************/

__host__ __device__ static inline void hydro_u(const hydro_t * hydro,
					       int index,
					       double u[3]) {
  assert(hydro);
  u[X] = hydro->u->data[addr_rank1(hydro->nsite, 3, index, X)];
  u[Y] = hydro->u->data[addr_rank1(hydro->nsite, 3, index, Y)];
  u[Z] = hydro->u->data[addr_rank1(hydro->nsite, 3, index, Z)];
  return;
}

/*****************************************************************************
 *
 *  hydro_f_local_set
 *
 *****************************************************************************/

__host__ __device__ static inline void hydro_f_local_set(hydro_t * hydro,
							 int index,
							 const double f[3]) {
  assert(hydro);
  hydro->force->data[addr_rank1(hydro->nsite, 3, index, X)] = f[X];
  hydro->force->data[addr_rank1(hydro->nsite, 3, index, Y)] = f[Y];
  hydro->force->data[addr_rank1(hydro->nsite, 3, index, Z)] = f[Z];

  return;
}

/*****************************************************************************
 *
 *  hydro_f_local
 *
 *****************************************************************************/

__host__ __device__ static inline void hydro_f_local(const hydro_t * hydro,
						     int index,
						     double force[3]) {
  assert(hydro);

  force[X] = hydro->force->data[addr_rank1(hydro->nsite, 3, index, X)];
  force[Y] = hydro->force->data[addr_rank1(hydro->nsite, 3, index, Y)];
  force[Z] = hydro->force->data[addr_rank1(hydro->nsite, 3, index, Z)];

  return;
}

/*****************************************************************************
 *
 *  hydro_f_local_add
 *
 *  Accumulate (repeat, accumulate) the fluid force at site index.
 *
 *****************************************************************************/

__host__ __device__ static inline void hydro_f_local_add(hydro_t * hydro,
							 int index,
							 const double f[3]) {
  assert(hydro);

  hydro->force->data[addr_rank1(hydro->nsite, 3, index, X)] += f[X]; 
  hydro->force->data[addr_rank1(hydro->nsite, 3, index, Y)] += f[Y]; 
  hydro->force->data[addr_rank1(hydro->nsite, 3, index, Z)] += f[Z]; 

  return;
}

/*****************************************************************************
 *
 *  hydro_rho_set
 *
 *****************************************************************************/

__host__ __device__ static inline void hydro_rho_set(hydro_t * hydro,
						     int index,
						     double rho) {
  assert(hydro);

  hydro->rho->data[addr_rank0(hydro->nsite, index)] = rho;

  return;
}

/*****************************************************************************
 *
 *  hydro_rho
 *
 *****************************************************************************/

__host__ __device__ static inline void hydro_rho(const hydro_t * hydro,
						 int index,
						 double * rho) {
  assert(hydro);
  assert(rho);

  *rho = hydro->rho->data[addr_rank0(hydro->nsite, index)];

  return;
}

#endif
