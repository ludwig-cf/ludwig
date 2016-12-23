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

#ifndef NDEBUG

#define assert_valid_index(lineno, file, nlen, index) \
  do {						       \
    if (index < 0 || index >= nlen) {		       \
      printf("%s (%s = %d, %s = %d), file %s, line %d\n", \
	      "Bad array index", #nlen, nlen, #index, index, file, lineno); \
      assert(index > 0 && index < nlen); \
    }		 \
  } while(0)

/*****************************************************************************
 *
 *  forward_addr_rank0_assert
 *
 *****************************************************************************/

__host__ __target__
int forward_addr_rank0_assert(int lineno, const char * file,
			      int nsites, int index) {

  assert_valid_index(lineno, file, nsites, index);

  return index;
}

/*****************************************************************************
 *
 *  forward_addr_rank1_assert
 *
 *****************************************************************************/

__host__ __target__
int forward_addr_rank1_assert(int lineno, const char * file,
			      int nsites, int na, int index, int ia) {

  assert_valid_index(lineno, file, nsites, index);
  assert_valid_index(lineno, file, na, ia);

  return na*index + ia;
}

/*****************************************************************************
 *
 *  forward_addr_rank2_assert
 *
 *****************************************************************************/

__host__ __target__
int forward_addr_rank2_assert(int lineno, const char * file,
			      int nsites, int na, int nb,
			      int index, int ia, int ib) {

  assert_valid_index(lineno, file, nsites, index);
  assert_valid_index(lineno, file, na, ia);
  assert_valid_index(lineno, file, nb, ib);

  return na*nb*index + nb*ia + ib;
}

/*****************************************************************************
 *
 *  forward_addr_rank3_assert
 *
 *****************************************************************************/

__host__ __target__
int forward_addr_rank3_assert(int lineno, const char * file,
			      int nsites, int na, int nb, int nc,
			      int index, int ia, int ib, int ic) {

  assert_valid_index(lineno, file, nsites, index);
  assert_valid_index(lineno, file, na, ia);
  assert_valid_index(lineno, file, nb, ib);
  assert_valid_index(lineno, file, nc, ic);

  return na*nb*nc*index + nb*nc*ia + nc*ib + ic;
}

/*****************************************************************************
 *
 *  forward_addr_rank4_assert
 *
 *****************************************************************************/

__host__ __target__
int forward_addr_rank4_assert(int lineno, const char * file,
			      int nsites, int na, int nb, int nc, int nd,
			      int index, int ia, int ib, int ic, int id) {

  assert_valid_index(lineno, file, nsites, index);
  assert_valid_index(lineno, file, na, ia);
  assert_valid_index(lineno, file, nb, ib);
  assert_valid_index(lineno, file, nc, ic);
  assert_valid_index(lineno, file, nd, id);

  return (na*nb*nc*nd*index + nb*nc*nd*ia + nc*nd*ib + nd*ic + id);
}

/*****************************************************************************
 *
 *  reverse_addr_rank0_assert
 *
 *****************************************************************************/

__host__ __target__
int reverse_addr_rank0_assert(int lineno, const char * file,
			      int nsites, int index) {

  assert_valid_index(lineno, file, nsites, index);

  return index;
}

/*****************************************************************************
 *
 *  reverse_addr_rank1_assert
 *
 *****************************************************************************/

__host__ __target__
int reverse_addr_rank1_assert(int lineno, const char * file,
			      int nsites, int na, int index, int ia) {

  assert_valid_index(lineno, file, nsites, index);
  assert_valid_index(lineno, file, na, ia);

  return nsites*ia + index;
}

/*****************************************************************************
 *
 *  reverse_addr_rank2_assert
 *
 *****************************************************************************/

__host__ __target__
int reverse_addr_rank2_assert(int lineno, const char * file,
			      int nsites, int na, int nb,
			      int index, int ia, int ib) {

  assert_valid_index(lineno, file, nsites, index);
  assert_valid_index(lineno, file, na, ia);
  assert_valid_index(lineno, file, nb, ib);

  return nb*nsites*ia + nsites*ib + index;
}

/*****************************************************************************
 *
 *  reverse_addr_rank3_assert
 *
 *****************************************************************************/

__host__ __target__
int reverse_addr_rank3_assert(int lineno, const char * file,
			      int nsites, int na, int nb, int nc,
			      int index, int ia, int ib, int ic) {

  assert_valid_index(lineno, file, nsites, index);
  assert_valid_index(lineno, file, na, ia);
  assert_valid_index(lineno, file, nb, ib);
  assert_valid_index(lineno, file, nc, ic);

  return nb*nc*nsites*ia + nc*nsites*ib + nsites*ic + index;
}

/*****************************************************************************
 *
 *  reverse_addr_rank4_assert
 *
 *****************************************************************************/

__host__ __target__
int reverse_addr_rank4_assert(int lineno, const char * file,
			      int nsites, int na, int nb, int nc, int nd,
			      int index, int ia, int ib, int ic, int id) {

  assert_valid_index(lineno, file, nsites, index);
  assert_valid_index(lineno, file, na, ia);
  assert_valid_index(lineno, file, nb, ib);
  assert_valid_index(lineno, file, nc, ic);
  assert_valid_index(lineno, file, nd, id);

  return nb*nc*nd*nsites*ia + nc*nd*nsites*ib + nd*nsites*ic + nsites*id + index;
}

#endif /* NDEBUG */

/*****************************************************************************
 *
 *  mem_addr_rank0
 *
 *****************************************************************************/

__host__ __target__
int mem_addr_rank0(int nsites, int index) {

  return addr_rank0(nsites, index);
}

/*****************************************************************************
 *
 *  mem_addr_rank1
 *
 *****************************************************************************/

__host__ __target__
int mem_addr_rank1(int nsites, int na, int index, int ia) {

  return addr_rank1(nsites, na, index, ia);
}

/*****************************************************************************
 *
 *  mem_addr_rank2
 *
 *****************************************************************************/

__host__ __target__
int mem_addr_rank2(int nsites, int na, int nb, int index, int ia, int ib) {

  return addr_rank2(nsites, na, nb, index, ia, ib);
}
