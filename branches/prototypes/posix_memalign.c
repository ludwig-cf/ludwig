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
