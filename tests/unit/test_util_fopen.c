/*****************************************************************************
 *
 *  test_util_fopen.c
 *
 *  Check the wrapper is working properly.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "util_fopen.h"

int test_util_fopen_default_umask(void);
int test_util_fopen(void);

/*****************************************************************************
 *
 *  test_util_fopen_suite
 *
 *****************************************************************************/

int test_util_fopen_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_util_fopen_default_umask();

  /* All actual file operations at root to prevent collisions */

  if (pe_mpi_rank(pe) == 0) test_util_fopen();

  pe_info(pe, "PASS     ./unit/test_util_fopen\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_util_fopen_default_umask
 *
 *****************************************************************************/

int test_util_fopen_default_umask(void) {

  int umask = util_fopen_default_umask();

  assert(umask == 0600);

  return (umask - 0600);
}

/*****************************************************************************
 *
 *  test_util_fopen
 *
 *****************************************************************************/

int test_util_fopen(void) {

  int ifail = 0;
  const char * path = "util_fopen_test_file.txt";

  {
    /* Write a new file ... */
    const char * mode = "w";
    FILE * fp = util_fopen(path, mode);
    if (ferror(fp)) {
      ifail = -1;
    }
    else {
      fprintf(fp, "text file\n");
      fclose(fp);
    }
  }

  {
    /* Read the file we just wrote ... */
    const char * mode = "r";
    FILE * fp = util_fopen(path, mode);
    if (ferror(fp)) {
      ifail = -1;
    }
    else {
      char buf[BUFSIZ] = {0};
      fgets(buf, BUFSIZ-1, fp);
      fclose(fp);
    }
  }

  {
    /* Append to same file ... */
    const char * mode = "a";
    FILE * fp = util_fopen(path, mode);
    if (ferror(fp)) {
      ifail = -1;
    }
    else {
      fprintf(fp, "another text line\n");
      fclose(fp);
    }
  }

  {
    /* And check remaining mode flags ... */
    const char * mode = "a+b";
    FILE * fp = util_fopen(path, mode);
    if (ferror(fp)) {
      ifail = -1;
    }
    else {
      fprintf(fp, "Final text line\n");
      fclose(fp);
    }
  }

  if (ifail == 0) remove(path);
  
  assert(ifail == 0);

  return ifail;
}
