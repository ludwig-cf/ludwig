/*****************************************************************************
 *
 *  memory.c
 *
 *  Array indexing functions.
 *
 *  We also include a wrapper for posix_memalign(), which is ...
 *
 *  #include <stdlib.h>
 *
 *  int posix_memalign(void ** memptr, size_t alignment, size_t size);
 *
 *  alignment must be (2^n)*sizeof(void *)
 *
 *  Returns
 *  0 success
 *  EINVAL alignment not (2^n)*sizeof(void *)
 *  ENOMEM memory not available
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2016-2017 The University of Edinbrugh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

__host__ __device__
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

__host__ __device__
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

__host__ __device__
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

__host__ __device__
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

__host__ __device__
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

__host__ __device__
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

__host__ __device__
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

__host__ __device__
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

__host__ __device__
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

__host__ __device__
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

__host__ __device__
int mem_addr_rank0(int nsites, int index) {

  return addr_rank0(nsites, index);
}

/*****************************************************************************
 *
 *  mem_addr_rank1
 *
 *****************************************************************************/

__host__ __device__
int mem_addr_rank1(int nsites, int na, int index, int ia) {

  return addr_rank1(nsites, na, index, ia);
}

/*****************************************************************************
 *
 *  mem_addr_rank2
 *
 *****************************************************************************/

__host__ __device__
int mem_addr_rank2(int nsites, int na, int nb, int index, int ia, int ib) {

  return addr_rank2(nsites, na, nb, index, ia, ib);
}

/*****************************************************************************
 *
 *  mem_aligned_malloc
 *
 *  A wrapper to posix_memalign() returning NULL if not successful.
 *
 *  Note: to detect (2^n)
 *  Eg. 8 = 1000 and 8-1 = 7 = 0111 so (1000 & 0111) = 0000
 *
 *  May be released via free().
 *
 *****************************************************************************/

void * mem_aligned_malloc(size_t alignment, size_t size) {

  int ifail;
  void * p;

  /* We assume these two assertions are sufficient to meet the
   * conditions on alignment ... */
  assert(alignment % sizeof(void *) == 0);
  assert((alignment & (alignment-1)) == 0);

  ifail = posix_memalign(&p, alignment, size);
  if (ifail) p = NULL;

  return p;
}

/*****************************************************************************
 *
 *  mem_aligned_calloc
 *
 *  Follows calloc() but with aligned allocation via mem_aligned_malloc().
 *
 *****************************************************************************/

void * mem_aligned_calloc(size_t alignment, size_t count, size_t size) {

  int ifail;
  void * p;

  ifail = posix_memalign(&p, alignment, count*size);
  if (ifail == 0) {
    memset(p, 0, count*size);
  }
  else {
    p = NULL;
  }

  return p;
}

/*****************************************************************************
 *
 *  mem_aligned_realloc
 *
 *  Follows realloc() but with aligned allocation via mem_aliened_malloc().
 *
 *  If size is greater than the existing size, the new content is undefined.
 *  If not enough memory, leave old memory alone and return NULL.
 *  If new size is 0, behaves like malloc().
 *
 *  As standard is silent on the alignment properties of realloc()
 *  always allocate a new block and copy.
 *
 *  We assume a copy of size size can be made.
 *
 *****************************************************************************/

void * mem_aligned_realloc(void * ptr, size_t alignment, size_t size) {

  void * p = NULL;

  if (ptr == NULL) {
    return mem_aligned_malloc(alignment, size);
  }
  else {
    p = mem_aligned_malloc(alignment, size);
    if (p) {
      memcpy(p, ptr, size);
      free(ptr);
    }
  }

  return p;
}
