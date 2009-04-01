/*****************************************************************************
 *
 *  test_assumptions.c
 *
 *  Test basic model assumptions, portability issues.
 *
 *  $Id: test_assumptions.c,v 1.5 2009-04-01 08:33:50 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2008 The University of Edinburgh 
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "util.h"

int main(int argc, char ** argv) {

  int n;
  int * p_int;

  printf("Testing assumptions...\n");

  /* All integers in the code should be declared 'int', which
   * we expect to be (at the least) 4 bytes. */

  printf("Checking sizeof(int) is 4 bytes... ");
  assert(sizeof(int) == 4);
  printf("yes\n");

  printf("Checking sizeof(long int) is >= 4 bytes... ");
  assert(sizeof(long int) >= 4);
  printf("yes (%ld bytes)\n", sizeof(long int));

  /* All floating point types in the code should be double,
   * which must be 8 bytes. */

  printf("Checking sizeof(float) is 4 bytes... ");
  assert(sizeof(float) == 4);
  printf("yes\n");

  printf("Checking sizeof(double) is 8 bytes... ");
  assert(sizeof(double) == 8);
  printf("yes\n");

  printf("Checking sizeof(void *) ...\n");
  printf("%d\n", sizeof(void *));

  printf("Checking FILENAME_MAX >= 128 characters ... ");
  assert(FILENAME_MAX >= 128);
  printf("yes (%d characters)\n", FILENAME_MAX);
  printf("Checking BUFSIZ... (%d bytes)\n", BUFSIZ);

  /* See what happens to zero size allocation */

  n = 0;
  p_int = (int *) malloc(n*sizeof(int));

  if (p_int == NULL) {
    printf("malloc(0) returns a NULL pointer\n");
  }
  else {
    printf("malloc(0) returns non NULL pointer\n");
    free(p_int);
  }


  /* Information */
  printf("Language\n");
  printf("__STC__ = %d\n", __STDC__);
#if (__STDC_VERSION__ >= 199901)
  printf("__STDC_VERSION__ = %ld\n", __STDC_VERSION__);
#endif
  printf("__DATE__ is %s\n", __DATE__);
  printf("__TIME__ is %s\n", __TIME__);
  printf("__FILE__ is %s\n", __FILE__);
  printf("__LINE__ is %d\n", __LINE__);

  printf("Host appears to be %s-endian\n", is_bigendian() ? "big" : "little");

  printf("All assumptions ok!\n");

  return 0;
}
