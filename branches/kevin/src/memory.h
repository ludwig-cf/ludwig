/*****************************************************************************
 *
 *  memory.h
 *
 *  Memory access descriptions.
 *
 *****************************************************************************/

#include <assert.h>

#define nsimd 2

/* Interface */

/*

For non-vectorised loops:

addr_rank1(nsites, na, index, ia)
addr_rank2(nsites, na, nb, index, ia, ib)
addr_rank3(nsites, na, nb, nc, index, ia, ib, ic)
...

For vectorised loops:

addrv_rank1(nsites, na, index, ia, iv)
addrv_rank2(nsites, na, nb, index, ia, ib, iv)
addrv_rank3(nsites, na, nb, nc, index, ia, ib, ic, iv)
...

*/

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
 * for (ithread = 0; ithread < nsites; ithread += nsimd) {
 *   baseindex = coords_index(ic, jc, kc);
 *   for (ia = 0; ia < na; ia++) {
 *     for (iv = 0; iv < nsimd; iv++) {
 *       array[addrv_rank1(nsites, na, baseindex, ia, iv)] = ...
 *     }
 *     ...
 */

/* Allocated as flat 1d arrays: */
/* Rank 1 array[nsites][na] */
/* Rank 2 array[nsites][na][nb] */
/* Rank 3 array[nsites][na][nb][nc] */

/* And effectively for SIMD short vectors we have: */
/* Rank 1 array[nsites/NSIMD][na][NSIMD] */
/* Rank 2 array[nsites/NSIMD][na][nb][NSIMD] */
/* Rank 3 array[nsites/NSIMD][na][nb][nc][NSIMD] */

/* Implementation */

#ifdef NDEBUG

#define forward_addr_rank1(nsites, na, index, ia) \
  ( (na)*(index) + (ia) )

#define forward_addr_rank2(nsites, na, nb, index, ia, ib) \
  ( (na)*(nb)*(index) + (nb)*(ia) + (ib) )

#define forward_addr_rank3(nsites, na, nb, nc, index, ia, ib, ic) \
  ( (na)*(nb)*(nc)*(index) + (nb)*(nc)*(ia)  + (nc)*(ib) + (ic))

#define forward_addr_rank4(nsites, na, nb, nc, nd, index, ia, ib, ic, id) \
  ( (na)*(nb)*(nc)*(nd)*(index) + (nb)*(nc)*(nd)*(ia) + (nc)*(nd)*(ib) + (nd)*(ic) + (id) )

#else

int forward_addr_rank1(int nsites, int na, int index, int ia) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && ia < na);

  return na*index + ia;
}

int forward_addr_rank2(int nsites, int na, int nb,
		       int index, int ia, int ib) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && ia < na);
  assert(ib >= 0    && ib < nb);

  return na*nb*index + nb*ia + ib;
}

int forward_addr_rank3(int nsites, int na, int nb, int nc,
		       int index, int ia, int ib, int ic) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && ia < na);
  assert(ib >= 0    && ib < nb);
  assert(ic >= 0    && ic < nc);

  return na*nb*nc*index + nb*nc*ia + nc*ib + ic;
}

int forward_addr_rank4(int nsites, int na, int nb, int nc, int nd,
		       int index, int ia, int ib, int ic, int id) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && ia < na);
  assert(ib >= 0    && ib < nb);
  assert(ic >= 0    && ic < nc);
  assert(id >= 0    && id < nd);

  return (na*nb*nc*nd*index + nb*nc*nd*ia + nc*nd*ib + nd*ic + id);
}

#endif /* NDEBUG */

/* 'Reverse' or coallescing order */

/* Effectively, we have:
 *
 * Rank 1 array[na][nsites]
 * Rank 2 array[na][nb][nsites]
 * Rank 3 array[na][nb][nc][nsites] */

#ifdef NDEBUG

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

int reverse_addr_rank1(int nsites, int na, int index, int ia) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && ia < na);

  return nsites*ia + index;
}

int reverse_addr_rank2(int nsites, int na, int nb, int index, int ia, int ib) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && ia < na);
  assert(ib >= 0    && ib < nb);

  return nb*nsites*ia + nsites*ib + index;
}

int reverse_addr_rank3(int nsites, int na, int nb, int nc,
		       int index, int ia, int ib, int ic) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && index < na);
  assert(ib >= 0    && index < nb);
  assert(ic >= 0    && index < nc);

  return nb*nc*nsites*ia + nc*nsites*ib + nsites*ic + index;
}

int reverse_addr_rank4(int nsites, int na, int nb, int nc, int nd,
		       int index, int ia, int ib, int ic, int id) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && index < na);
  assert(ib >= 0    && index < nb);
  assert(ic >= 0    && index < nc);
  assert(id >= 0    && index < nd);

  return nb*nc*nd*nsites*ia + nc*nd*nsites*ib + nd*nsites*ic + nsites*id + index;
}
#endif


#ifndef ADDR_MODEL_R
#define base_addr_rank1 forward_addr_rank1
#define base_addr_rank2 forward_addr_rank2
#define base_addr_rank3 forward_addr_rank3
#define base_addr_rank4 forward_addr_rank4
#else /* REVERSE */
#define base_addr_rank1 reverse_addr_rank1
#define base_addr_rank2 reverse_addr_rank2
#define base_addr_rank3 reverse_addr_rank3
#define base_addr_rank4 reverse_addr_rank4
#endif

/* Non-vectorised loops. */
/* We simulate an innermost vector loop by arithmetic based
 * on the coordinate index, which is expected to run normally
 * from 0 ... nites-1. The 'dummy' vector loop index is ... */

#define pseudo_iv(index) ( (index) - (((index)/nsimd)*nsimd) )


/* Macro definitions for the interface */

#define addr_rank1(nsites, na, index, ia) \
  base_addr_rank2(nsites/nsimd, na, nsimd, (index)/nsimd, ia, pseudo_iv(index))

#define addr_rank2(nsites, na, nb, index, ia, ib) \
  base_addr_rank3(nsites/nsimd, na, nb, nsimd, (index)/nsimd, ia, ib, pseudo_iv(index))

#define addr_rank3(nsites, na, nb, nc, index, ia, ib, ic) \
  base_addr_rank4(nsites/nsimd, na, nb, nc, nsimd, (index)/nsimd, ia, ib, ic, pseudo_iv(index))


#define addrv_rank1(nsites, na, index, ia, iv) \
  base_addr_rank2(nsites/nsimd, na, nsimd, (index)/nsimd, ia, iv)

#define addrv_rank2(nsites, na, nb, index, ia, ib, iv) \
  base_addr_rank3(nsites/nsimd, na, nb, nsimd, (index)/nsimd, ia, ib, iv)

#define addrv_rank3(nsites, na, nb, nc, index, ia, ib, ic, iv) \
  base_addr_rank4(nsites/nsimd, na, nb, nc, nsimd, (index)/nsimd, ia, ib, ic, iv)

