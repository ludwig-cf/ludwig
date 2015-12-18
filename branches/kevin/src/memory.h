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

For vectorised loops:

addrv_rank1(nsites, na, index, ia, iv)
addrv_rank2(nsites, na, nb, index, ia, ib, iv)
addrv_rank3(nsites, na, nb, nc, index, ia, ib, ic, iv)

*/

/* So, in all situations, the following forms should be
 *  - consistent
 *  - access memory in the appropriate order in the vectorised
 *    target loop
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

#define base_addr_rank1(nsites, na, index, ia) \
  ( (na)*(index) + (ia) )

#define base_addr_rank2(nsites, na, nb, index, ia, ib) \
  ( (na)*(nb)*(index) + (nb)*(ia) + (ib) )

#define base_addr_rank3(nsites, na, nb, nc, index, ia, ib, ic) \
  ( (na)*(nb)*(nc)*(index) + (nb)*(nc)*(ia)  + (nc)*(ib) + (ic))

#else

int base_addr_rank1(int nsites, int na, int index, int ia) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && ia < na);

  return na*index + ia;
}

int base_addr_rank2(int nsites, int na, int nb,
		    int index, int ia, int ib) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && ia < na);
  assert(ib >= 0    && ib < nb);

  return na*nb*index + nb*ia + ib;
}

int base_addr_rank3(int nsites, int na, int nb, int nc,
		    int index, int ia, int ib, int ic) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && ia < na);
  assert(ib >= 0    && ib < nb);
  assert(ic >= 0    && ic < nc);

  return na*nb*nc*index + nb*nc*ia + nc*ib + ic;
}

#endif /* NDEBUG */


#define pseudo_iv(index) ( (index) - (((index)/nsimd)*nsimd) )

#define model_addr_rank1(nsites, na, index, ia) \
  base_addr_rank2(nsites/nsimd, na, nsimd, index/nsimd, ia, pseudo_iv(index))

#define model_addrv_rank1(nsites, na, index, ia, iv) \
  base_addr_rank2(nsites/nsimd, na, nsimd, index/nsimd, ia, iv)

#define model_addr_rank2(nsites, na, nb, index, ia, ib) \
  model_addr_rank3(nsites/nsimd, na, nb, nsimd, index, ia, ib, pseudo_iv(index))
