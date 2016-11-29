/*
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
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {

  int ia;
  int ifail;
  double * p = NULL;

  for (ia = 0; ia < 10; ia++) {
    ifail = posix_memalign((void **) &p, 64, 128);
    assert(ifail == 0);

    printf("p is %p\n", p);
  }

  free(p);

  return 0;
}

/*****************************************************************************
 *
 *  util_aligned_malloc
 *
 *  A wrapper to posix_memalign() returning NULL if not successful.
 *
 *  Note: to detect (2^n)
 *  Eg. 8 = 1000 and 8-1 = 7 = 0111 so (1000 & 0111) = 0000
 *
 *  May be released via free().
 *
 *****************************************************************************/

void * util_aligned_malloc(size_t alignment, size_t size) {

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
 *  util_aligned_calloc
 *
 *  Follows calloc() but with aligned allocation via util_aligned_malloc().
 *
 *****************************************************************************/

void * util_aligned_calloc(size_t alignment, size_t count, size_t size) {

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
 *  util_aligned_realloc
 *
 *  Follows realloc() but with aligned allocation via util_aliened_malloc().
 *
 *  If size is greater than the existing size, the new content is undefined.
 *  If not enough memory, leave old memory alone and return NULL.
 *  If new size is 0, behaves like malloc().
 *
 *  As standard is silent on the alignment properties of realloc()
 *  always allocate a new block and copy.
 *
 *****************************************************************************/

void * util_aligned_realloc(void * ptr, size_t alignment, size_t size) {

  void * p = NULL;

  if (ptr == NULL) {
    return util_aligned_malloc(alignment, size);
  }
  else {
    p = util_aligned_malloc(alignment, size);
    if (p) {
      memcpy(p, ptr, size);
      free(ptr);
    }
  }

  return p;
}
