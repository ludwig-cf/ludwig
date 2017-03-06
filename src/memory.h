/*****************************************************************************
 *
 *  memory.h
 *
 *  Memory access descriptions of scalar, vector, tensor, ... fields.
 *
 *  Meaningful Choices:
 *
 *    Forward/Reverse   Blocking   VECTOR LENGTH         PREPROCESSOR
 * 1. Forward (AOS)     No         1                     
 * 2. Forward           No         N                     -DVVL=N
 * 3. Forward           Yes        1  -> Same as 1
 * 4. Forward           Yes        N                     -DVVL=N -DADDR_AOSOA
 * 5. Reverse (SOA)     No         1                     -DADDR_SOA
 * 6. Reverse           No         N  Not tested.
 * 7. Revsrse           Yes        1  -> Same as 5
 * 8. Reverse           Yes        N  Not implemented.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 * 
 *  (c) 2016-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef MEMORY_MODEL_H
#define MEMORY_MODEL_H

#include <assert.h>
#include "pe.h"

/* The target SIMD vector length */

#define NSIMDVL VVL

/* Interface */

/*

addr_rank0(nsites, index);
addr_rank1(nsites, na, index, ia)
addr_rank2(nsites, na, nb, index, ia, ib)
addr_rank3(nsites, na, nb, nc, index, ia, ib, ic)
...

*/

/* Interface not dependent on preprocessor directives at compile time.
   For non-performance critical use. */

__host__ __target__
int mem_addr_rank0(int nsites, int index);
__host__ __target__
int mem_addr_rank1(int nsites, int na, int index, int ia);
__host__ __target__
int mem_addr_rank2(int nsites, int na, int nb, int index, int ia, int ib);

/* End of interface */

/* So, in all situations, the following forms should be
 *  1. consistent
 *  2. access memory in the appropriate order in the vectorised
 *     target loop
 *
 * A "host loop" construct accessing a rank1 array[nsites][na]
 *
 * for (ic = 1; ic <= nlocal[X]; ic++) {
 *   for (jc = 1; jc <= nlocal[Y]; jc++) {
 *     for (kc = 1; kc <= nlocal[Z]; kc++) {
 *       index = coords_index(ic, jc, kc);
 *       for (ia = 0; ia < na; ia++) {
 *         array[addr_rank1(nsites, na, index, ia)] = ...
 *       }
 *       ...
 *
 * A "target loop" without vectorisation
 *
 * for (ithread = 0; ithread < nsites; ithread++) {
 *   index = ithread;
 *   for (ia = 0; ia < na; ia++) {
 *     array[addr_rank1(nsites, na, index, ia)] = ...
 *   }
 *
 * A "target loop" with an explicit innermost vector loop
 *
 * for (ithread = 0; ithread < nsites; ithread += NSIMDVL) {
 *   baseindex = coords_index(ic, jc, kc);
 *   for (ia = 0; ia < na; ia++) {
 *     for (iv = 0; iv < NSIMDVL; iv++) {
 *       array[addr_rank1(nsites, na, baseindex + iv, ia)] = ...
 *     }
 *     ...
 */

/* Allocated as flat 1d arrays: */
/* Rank 1 array[nsites][na] */
/* Rank 2 array[nsites][na][nb] */
/* Rank 3 array[nsites][na][nb][nc] */

/* And effectively for SIMD short vectors we have: */
/* Rank 1 array[nsites/NSIMDVL][na][NSIMDVL] */
/* Rank 2 array[nsites/NSIMDVL][na][nb][NSIMDVL] */
/* Rank 3 array[nsites/NSIMDVL][na][nb][nc][NSIMDVL] */

/* Implementation */

typedef enum data_model_enum_type {DATA_MODEL_AOS,
				   DATA_MODEL_SOA,
				   DATA_MODEL_AOSOA}
  data_model_enum_t;

#ifdef NDEBUG

/* Forward, or AOS order. */

#define forward_addr_rank0(nsites, index) (index)

#define forward_addr_rank1(nsites, na, index, ia) \
  ( (na)*(index) + (ia) )

#define forward_addr_rank2(nsites, na, nb, index, ia, ib) \
  ( (na)*(nb)*(index) + (nb)*(ia) + (ib) )

#define forward_addr_rank3(nsites, na, nb, nc, index, ia, ib, ic) \
  ( (na)*(nb)*(nc)*(index) + (nb)*(nc)*(ia)  + (nc)*(ib) + (ic))

#define forward_addr_rank4(nsites, na, nb, nc, nd, index, ia, ib, ic, id) \
  ( (na)*(nb)*(nc)*(nd)*(index) + (nb)*(nc)*(nd)*(ia) + (nc)*(nd)*(ib) + (nd)*(ic) + (id) )

#else

__host__ __target__
int forward_addr_rank0_assert(int line, const char * file,
			      int nsites, int index);
__host__ __target__
int forward_addr_rank1_assert(int line, const char * file,
			      int nsites, int na, int index, int ia);
__host__ __target__
int forward_addr_rank2_assert(int line, const char * file,
			      int nsites, int na, int nb, int index,
			      int ia, int ib);
__host__ __target__
int forward_addr_rank3_assert(int line, const char * file,
			      int nsites, int na, int nb, int nc,
			      int index, int ia, int ib, int ic);
__host__ __target__
int forward_addr_rank4_assert(int line, const char * file,
			      int nsites, int na, int nb, int nc, int nd,
			      int index, int ia, int ib, int ic, int id);

#define forward_addr_rank0(...) \
  forward_addr_rank0_assert(__LINE__, __FILE__, __VA_ARGS__) 
#define forward_addr_rank1(...) \
  forward_addr_rank1_assert(__LINE__, __FILE__, __VA_ARGS__) 
#define forward_addr_rank2(...) \
  forward_addr_rank2_assert(__LINE__, __FILE__, __VA_ARGS__)
#define forward_addr_rank3(...) \
  forward_addr_rank3_assert(__LINE__, __FILE__, __VA_ARGS__)
#define forward_addr_rank4(...) \
  forward_addr_rank4_assert(__LINE__, __FILE__, __VA_ARGS__)

#endif /* NDEBUG */

/* 'Reverse' or SOA, or coallescing order */

/* Effectively, we have:
 *
 * Rank 1 array[na][nsites]
 * Rank 2 array[na][nb][nsites]
 * Rank 3 array[na][nb][nc][nsites] */

#ifdef NDEBUG

#define reverse_addr_rank0(nsites, index) (index)

#define reverse_addr_rank1(nsites, na, index, ia) \
  ( (nsites)*(ia) + (index) )

#define reverse_addr_rank2(nsites, na, nb, index, ia, ib) \
  ( (nb)*(nsites)*(ia) + (nsites)*(ib) + (index) )

#define reverse_addr_rank3(nsites, na, nb, nc, index, ia, ib, ic)	\
  ( (nb)*(nc)*(nsites)*(ia) + (nc)*(nsites)*(ib) + (nsites)*(ic) + (index) )

#define reverse_addr_rank4(nsites, na, nb, nc, nd, index, ia, ib, ic, id) \
  ( (nb)*(nc)*(nd)*(nsites)*(ia) + (nc)*(nd)*(nsites)*(ib) + \
    (nd)*(nsites)*(ic) + (nsites)*(id) + (index) )

#else

__host__ __target__
int reverse_addr_rank0_assert(int line, const char * file,
			      int nsites, int index);
__host__ __target__
int reverse_addr_rank1_assert(int line, const char * file,
			      int nsites, int na, int index, int ia);
__host__ __target__
int reverse_addr_rank2_assert(int line, const char * file,
			      int nsites, int na, int nb,
			      int index, int ia, int ib);
__host__ __target__
int reverse_addr_rank3_assert(int line, const char * file,
			      int nsites, int na, int nb, int nc,
			      int index, int ia, int ib, int ic);
__host__ __target__
int reverse_addr_rank4_assert(int line, const char * file,
			      int nsites, int na, int nb, int nc, int nd,
			      int index, int ia, int ib, int ic, int id);

#define reverse_addr_rank0(...)	\
  reverse_addr_rank0_assert(__LINE__, __FILE__, __VA_ARGS__) 
#define reverse_addr_rank1(...) \
  reverse_addr_rank1_assert(__LINE__, __FILE__, __VA_ARGS__) 
#define reverse_addr_rank2(...) \
  reverse_addr_rank2_assert(__LINE__, __FILE__, __VA_ARGS__)
#define reverse_addr_rank3(...) \
  reverse_addr_rank3_assert(__LINE__, __FILE__, __VA_ARGS__)
#define reverse_addr_rank4(...) \
  reverse_addr_rank4_assert(__LINE__, __FILE__, __VA_ARGS__)

#endif

/* Here is the choise of direction */

#ifndef ADDR_SOA
#define base_addr_rank0 forward_addr_rank0
#define base_addr_rank1 forward_addr_rank1
#define base_addr_rank2 forward_addr_rank2
#define base_addr_rank3 forward_addr_rank3
#define base_addr_rank4 forward_addr_rank4
#define DATA_MODEL DATA_MODEL_AOS
#else /* then it is SOA */
#define base_addr_rank0 reverse_addr_rank0
#define base_addr_rank1 reverse_addr_rank1
#define base_addr_rank2 reverse_addr_rank2
#define base_addr_rank3 reverse_addr_rank3
#define base_addr_rank4 reverse_addr_rank4
#define DATA_MODEL DATA_MODEL_SOA
#endif


/* Macro definitions for the interface */

#ifndef ADDR_AOSOA

#define addr_rank0 base_addr_rank0
#define addr_rank1 base_addr_rank1
#define addr_rank2 base_addr_rank2
#define addr_rank3 base_addr_rank3

#else

#define DATA_MODEL DATA_MODEL_AOSOA

/* Blocked version. Block length always vector length. */

#define NVBLOCK NSIMDVL

/* We simulate an innermost vector loop by arithmetic based
 * on the coordinate index, which is expected to run normally
 * from 0 ... nites-1. The 'dummy' vector loop index is ... */

#define pseudo_iv(index) ( (index) - ((index)/NVBLOCK)*NVBLOCK )

#define addr_rank0(nsites, index) \
  base_addr_rank1((nsites)/NVBLOCK, NVBLOCK, (index)/NVBLOCK, pseudo_iv(index))

#define addr_rank1(nsites, na, index, ia) \
  base_addr_rank2((nsites)/NVBLOCK, na, NVBLOCK, (index)/NVBLOCK, ia, pseudo_iv(index))

#define addr_rank2(nsites, na, nb, index, ia, ib) \
  base_addr_rank3((nsites)/NVBLOCK, na, nb, NVBLOCK, (index)/NVBLOCK, ia, ib, pseudo_iv(index))

#define addr_rank3(nsites, na, nb, nc, index, ia, ib, ic) \
  base_addr_rank4((nsites)/NVBLOCK, na, nb, nc, NVBLOCK, (index)/NVBLOCK, ia, ib, ic, pseudo_iv(index))

#endif

/* Alignment */

#define MEM_PAGESIZE 4096
void * mem_aligned_malloc(size_t alignment, size_t size);
void * mem_aligned_calloc(size_t alignment, size_t count, size_t size);
void * mem_aligned_realloc(void * ptr, size_t alignment, size_t size);


#endif
