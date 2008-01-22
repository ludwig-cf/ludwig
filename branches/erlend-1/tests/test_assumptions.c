/*****************************************************************************
 *
 *  Test basic model assumptions, portability issues.
 *
 *  Look to include stuff which is possibly machine-dependent
 *
 * 
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <stdio.h>
#include <assert.h>

int main(int argc, char ** argv) {

  printf("Testing assumptions...\n");

  /* All integers in the code should be declared 'int', which
   * we expect to be (at the least) 4 bytes. */

  printf("Checking sizeof(int) is 4 bytes... ");
  assert(sizeof(int) == 4);
  printf("yes\n");

  printf("Checking sizeof(long int) is 4 bytes... ");
  assert(sizeof(long int) == 4);
  printf("yes\n");

#ifdef _TESTS_PLUS_KR_
  /* Note that long long requires K&R extensions to ANSI,
   * so this might break under strictly conformant ANSI
   * compiler. However, there are no long long in the code. */
  printf("Checking sizeof(long long int) is 8 bytes... ");
  assert(sizeof(long long int) == 8);
  printf("yes\n");
#endif

  /* All floating point types in the code should be double,
   * which must be 8 bytes. */

  printf("Checking sizeof(float) is 4 bytes... ");
  assert(sizeof(float) == 4);
  printf("yes\n");

  printf("Checking sizeof(double) is 8 bytes... ");
  assert(sizeof(double) == 8);
  printf("yes\n");

  printf("Checking FILENAME_MAX >= 128 characters ... ");
  assert(FILENAME_MAX >= 128);
  printf("yes\n");

  printf("All assumptions ok!\n");

  return 0;
}
