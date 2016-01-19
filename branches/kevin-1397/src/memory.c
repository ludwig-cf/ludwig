/*****************************************************************************
 *
 *  memory.c
 *
 *  Array indexing functions.
 *
 *
 *****************************************************************************/

#include <assert.h>
#include "memory.h"

#ifdef NDEBUG

int keep_linker_quiet(void) {
  return 1;
}

#else

/*****************************************************************************
 *
 *  forward_addr_rank1
 *
 *****************************************************************************/

inline int forward_addr_rank1(int nsites, int na, int index, int ia) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && ia < na);

  return na*index + ia;
}

/*****************************************************************************
 *
 *  forward_addr_rank2
 *
 *****************************************************************************/

inline int forward_addr_rank2(int nsites, int na, int nb,
			      int index, int ia, int ib) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && ia < na);
  assert(ib >= 0    && ib < nb);

  return na*nb*index + nb*ia + ib;
}

/*****************************************************************************
 *
 *  forward_addr_rank3
 *
 *****************************************************************************/

int forward_addr_rank3(int nsites, int na, int nb, int nc,
		       int index, int ia, int ib, int ic) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && ia < na);
  assert(ib >= 0    && ib < nb);
  assert(ic >= 0    && ic < nc);

  return na*nb*nc*index + nb*nc*ia + nc*ib + ic;
}

/*****************************************************************************
 *
 *  forward_addr_rank4
 *
 *****************************************************************************/

int forward_addr_rank4(int nsites, int na, int nb, int nc, int nd,
		       int index, int ia, int ib, int ic, int id) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && ia < na);
  assert(ib >= 0    && ib < nb);
  assert(ic >= 0    && ic < nc);
  assert(id >= 0    && id < nd);

  return (na*nb*nc*nd*index + nb*nc*nd*ia + nc*nd*ib + nd*ic + id);
}

/*****************************************************************************
 *
 *  reverse_addr_rank1
 *
 *****************************************************************************/

int reverse_addr_rank1(int nsites, int na, int index, int ia) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && ia < na);

  return nsites*ia + index;
}

/*****************************************************************************
 *
 *  reverse_addr_rank2
 *
 *****************************************************************************/

int reverse_addr_rank2(int nsites, int na, int nb, int index, int ia, int ib) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && ia < na);
  assert(ib >= 0    && ib < nb);

  return nb*nsites*ia + nsites*ib + index;
}

/*****************************************************************************
 *
 *  reverse_addr_rank3
 *
 *****************************************************************************/

int reverse_addr_rank3(int nsites, int na, int nb, int nc,
		       int index, int ia, int ib, int ic) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && index < na);
  assert(ib >= 0    && index < nb);
  assert(ic >= 0    && index < nc);

  return nb*nc*nsites*ia + nc*nsites*ib + nsites*ic + index;
}

/*****************************************************************************
 *
 *  reverse_addr_rank4
 *
 *****************************************************************************/

int reverse_addr_rank4(int nsites, int na, int nb, int nc, int nd,
		       int index, int ia, int ib, int ic, int id) {

  assert(index >= 0 && index < nsites);
  assert(ia >= 0    && index < na);
  assert(ib >= 0    && index < nb);
  assert(ic >= 0    && index < nc);
  assert(id >= 0    && index < nd);

  return nb*nc*nd*nsites*ia + nc*nd*nsites*ib + nd*nsites*ic + nsites*id + index;
}

#endif /* NDEBUG */
