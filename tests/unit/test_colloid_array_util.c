/*****************************************************************************
 *
 *  test_colloid_array_util.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2025-2026 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "colloids.h"             /* We need the complete type here */
#include "colloid_array_util.h"

int test_colloid_array_alloc(void);
int test_colloid_array_realloc(void);
int test_colloid_array_free(void);

int test_colloid_pointer_array_alloc(void);
int test_colloid_pointer_array_realloc(void);
int test_colloid_pointer_array_free(void);

/*****************************************************************************
 *
 *  test_colloid_array_util_suite
 *
 *****************************************************************************/

int test_colloid_array_util_suite(void) {

  int    ifail = 0;
  pe_t * pe    = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* If struct changes, the tests need updating... */
  assert(sizeof(colloid_array_t) == 16);
  assert(sizeof(colloid_pointer_array_t) == 16);

  test_colloid_array_alloc();
  test_colloid_array_realloc();
  test_colloid_array_free();

  test_colloid_pointer_array_alloc();
  test_colloid_pointer_array_realloc();
  test_colloid_pointer_array_free();

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_array_alloc
 *
 *  Split into two parts for standard, and managed, as the managed
 *  versions require separate device tests.
 *
 *****************************************************************************/

int test_colloid_array_alloc_host(void);
int test_colloid_array_alloc_managed(void);

int test_colloid_array_alloc(void) {

  test_colloid_array_alloc_host();
  test_colloid_array_alloc_managed();

  return 0;
}

/*****************************************************************************
 *
 *  test_colloid_array_realloc
 *
 *  This is split into two again.
 *
 *****************************************************************************/

int test_colloid_array_realloc_host(void);
int test_colloid_array_realloc_managed(void);

int test_colloid_array_realloc(void) {

  test_colloid_array_realloc_host();
  test_colloid_array_realloc_managed();

  return 0;
}

/*****************************************************************************
 *
 *  test_colloid_array_free
 *
 *  For completeness' sake.
 *
 *****************************************************************************/

int test_colloid_array_free(void) {

  int ifail = 0;

  /* Host */
  {
    int managed = 0;
    int ntotal  = 1;

    colloid_array_t buf = {};

    ifail = colloid_array_alloc(managed, ntotal, &buf);
    assert(ifail == 0);

    colloid_array_free(&buf);
    assert(buf.managed == 0);
    assert(buf.ntotal  == 0);
    assert(buf.data    == NULL);
  }

  /* Managed */
  {
    int managed = 1;
    int ntotal  = 1;

    colloid_array_t buf = {};

    ifail = colloid_array_alloc(managed, ntotal, &buf);
    assert(ifail == 0);

    colloid_array_free(&buf);
    assert(buf.managed == 0);
    assert(buf.ntotal  == 0);
    assert(buf.data    == NULL);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_pointer_array_alloc
 *
 *  Split into host/device parts
 *
 *****************************************************************************/

int test_colloid_pointer_array_alloc_host(void);
int test_colloid_pointer_array_alloc_managed(void);

int test_colloid_pointer_array_alloc(void) {

  test_colloid_pointer_array_alloc_host();
  test_colloid_pointer_array_alloc_managed();

  return 0;
}

/*****************************************************************************
 *
 *  test_colloid_pointer_array_realloc
 *
 *****************************************************************************/

int test_colloid_pointer_array_realloc_host(void);
int test_colloid_pointer_array_realloc_managed(void);

int test_colloid_pointer_array_realloc(void) {

  test_colloid_pointer_array_realloc_host();
  test_colloid_pointer_array_realloc_managed();

  return 0;
}

/*****************************************************************************
 *
 *  test_colloid_array_alloc_host
 *
 *****************************************************************************/

int test_colloid_array_alloc_host(void) {

  int ifail   = 0;
  int managed = 0; /* all host allocations */

  /* Check zero-sized alloc fails elegantly ... */
  {
    int             ntotal = 0;
    colloid_array_t buf    = {};

    ifail = colloid_array_alloc(managed, ntotal, &buf);
    assert(ifail != 0);
  }

  /* Allocation should give us the right to access relevant elements */
  {
    int             ntotal = 19;
    colloid_array_t buf    = {};

    ifail = colloid_array_alloc(managed, ntotal, &buf);
    assert(ifail == 0);

    assert(buf.managed == managed);
    assert(buf.ntotal  == ntotal);
    assert(buf.data != NULL);

    for (int i = 0; i < buf.ntotal; i++) {
      buf.data[i] = (colloid_state_t) {};
    }

    colloid_array_free(&buf);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_array_alloc_managed
 *
 *  There are two kernels:
 *
 *****************************************************************************/

__global__ void kernel1(int ntotal, colloid_array_t buf) {

  int i = blockIdx.x; /* Index via block (all threads) */

  assert(buf.managed == 1);
  assert(buf.ntotal  == ntotal);
  assert(buf.data    != NULL);

  if (i >= ntotal) {
    printf("fail for all threads: blockIdx.x %d\n", i);
    assert(0);
  }

  if (buf.data[i].index != 1 + i) {
    printf("fail at %d\n", i);
    assert(0);
  }

  assert(buf.data[i].index == 1 + i);

  return;
}

__global__ void kernel2(int ntotal, colloid_array_t buf) {

  int i = blockIdx.x; /* Index via block (all threads) */

  assert(buf.managed == 1);
  assert(buf.ntotal  == ntotal);
  assert(buf.data    != NULL);

  if (i >= ntotal) {
    printf("fail for all threads: blockIdx.x %d\n", i);
    assert(0);
  }

  if (threadIdx.x == 0) {
    buf.data[i].index = 1 + i;
  }

  return;
}

/*****************************************************************************
 *
 *  test_colloid_array_alloc_managed
 *
 *  Driver.
 *
 *****************************************************************************/

int test_colloid_array_alloc_managed(void) {

  int ifail   = 0;
  int managed = 1; /* managed allocations */

  /* Host assignment */
  {
    int             ntotal = 32;
    colloid_array_t buf    = {};

    ifail = colloid_array_alloc(managed, ntotal, &buf);
    assert(ifail == 0);
    assert(buf.managed == managed);
    assert(buf.ntotal  == ntotal);
    assert(buf.data    != NULL);

    /* We have the right to access these elements ... */
    for (int i = 0; i < buf.ntotal; i++) {
      buf.data[i]       = (colloid_state_t) {};
      buf.data[i].index = 1 + i;
    }

    /* Values should be reflected in a kernel */
    /* Run one block for each array element.  */
    {
      dim3 blocks  = {  1, 1, 1};
      dim3 threads = {128, 1, 1};

      blocks.x = ntotal;

      tdpLaunchKernel(kernel1, blocks, threads, 0, 0, ntotal, buf);
      tdpAssert(tdpStreamSynchronize(0));
    }

    colloid_array_free(&buf);
  }

  /* Kernel assignment */
  {
    int             ntotal = 129;
    colloid_array_t buf    = {};

    ifail = colloid_array_alloc(managed, ntotal, &buf);
    assert(ifail == 0);

    /* Kernel; again one block per element, */
    /* kernel2 assigns values */
    {
      dim3 blocks  = { 1, 1, 1};
      dim3 threads = {32, 1, 1};

      blocks.x = ntotal;

      tdpLaunchKernel(kernel2, blocks, threads, 0, 0, ntotal, buf);
      tdpAssert(tdpStreamSynchronize(0));
    }

    /* And check ... */
    for (int i = 0; i < buf.ntotal; i++) {
      if (buf.data[i].index != 1 + i) {
        ifail = -1;
      }
      assert(ifail == 0);
    }

    colloid_array_free(&buf);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_array_realloc_host
 *
 *****************************************************************************/

int test_colloid_array_realloc_host(void) {

  int ifail = 0;

  /* realloc with an empty object */
  {
    int             newtotal = 20;
    colloid_array_t buf      = {};

    ifail = colloid_array_realloc(newtotal, &buf);
    assert(ifail == 0);

    assert(buf.managed == 0);
    assert(buf.ntotal  == newtotal);
    assert(buf.data);

    /* Check elements can be accessed */
    for (int i = 0; i < buf.ntotal; i++) {
      buf.data[i] = (colloid_state_t) {};
    }

    colloid_array_free(&buf);
  }

  /* Check existing data is preserved */
  {
    int managed  = 0;
    int ntotal   = 10;
    int newtotal = 20;

    colloid_array_t buf = {};

    colloid_array_alloc(managed, ntotal, &buf);
    for (int i = 0; i < buf.ntotal; i++) {
      buf.data[i]       = (colloid_state_t) {};
      buf.data[i].index = 1 + i;
    }

    /* re-allocate */
    ifail = colloid_array_realloc(newtotal, &buf);
    assert(ifail == 0);
    assert(buf.managed == managed);
    assert(buf.ntotal == newtotal);
    assert(buf.data);

    /* Check existing data, and assess the new data. */
    for (int i = 0; i < buf.ntotal; i++) {
      if (i < ntotal) {
        if (buf.data[i].index != 1 + i) {
          ifail = -1;
        }
        assert(ifail == 0);
      }
      buf.data[i] = (colloid_state_t) {};
    }

    colloid_array_free(&buf);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_array_realloc_managed
 *
 *  Two kernels are required.
 *
 *****************************************************************************/

int test_colloid_array_realloc_managed(void) {

  int ifail = 0;

  /* Allocate and set some values using kernel2, then re-allocate,
   * set the additional values on the host, and check with kernel1. */
  {
    int managed  = 1;
    int ntotal   = 10;
    int newtotal = 20;

    colloid_array_t buf = {};

    ifail = colloid_array_alloc(managed, ntotal, &buf);

    {
      dim3 blocks  = { 1, 1, 1};
      dim3 threads = {32, 1, 1};

      blocks.x = ntotal;

      tdpLaunchKernel(kernel2, blocks, threads, 0, 0, ntotal, buf);
      tdpAssert(tdpStreamSynchronize(0));
    }

    /* Re-allocate */
    ifail = colloid_array_realloc(newtotal, &buf);
    assert(ifail == 0);

    assert(buf.managed == 1);
    assert(buf.ntotal  == newtotal);
    assert(buf.data);

    /* set additional values ... */
    for (int i = ntotal; i < newtotal; i++) {
      buf.data[i]       = (colloid_state_t) {};
      buf.data[i].index = 1 + i;
    }

    /* Recheck */
    {
      dim3 blocks  = { 1, 1, 1};
      dim3 threads = {32, 1, 1};

      blocks.x = newtotal;

      tdpLaunchKernel(kernel1, blocks, threads, 0, 0, newtotal, buf);
      tdpAssert(tdpStreamSynchronize(0));
    }

    colloid_array_free(&buf);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_pointer_array_alloc_host
 *
 *****************************************************************************/

int test_colloid_pointer_array_alloc_host(void) {

  int ifail = 0;

  /* Check failure */
  {
    int managed = 0;
    int ntotal  = 0;

    colloid_pointer_array_t array = {};

    ifail = colloid_pointer_array_alloc(managed, ntotal, &array);
    assert(ifail != 0);
  }

  /* Check allocation/assignment */
  {
    int managed = 0;
    int ntotal  = 32;

    colloid_pointer_array_t array = {};

    ifail = colloid_pointer_array_alloc(managed, ntotal, &array);
    assert(ifail == 0);
    assert(array.managed == managed);
    assert(array.ntotal  == ntotal);

    for (int n = 0; n < ntotal; n++) {
      array.colloid[n] = NULL;
    }

    colloid_pointer_array_free(&array);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_pointer_array_alloc_managed
 *
 *****************************************************************************/

__global__ void kernel3(int ntotal, colloid_pointer_array_t array) {

  int index = blockIdx.x;  /* Index by block (all threads) */

  assert(array.managed == 1);
  assert(array.ntotal  == ntotal);
  assert(array.colloid);

  if (array.colloid[index]->s.index != 1 + index) {
    printf("Fail at %2d\n", index);
    assert(0);
  }

  return;
}

__global__ void kernel4(int ntotal, colloid_pointer_array_t array) {

  int index = blockIdx.x;   /* Index by block (all threads) */

  assert(array.managed == 1);
  assert(array.ntotal  == ntotal);
  assert(array.colloid != NULL);

  assert(gridDim.x == ntotal);

  if (threadIdx.x == 0) {
    array.colloid[index]->s.index = 1 + index;
  }

  return;
}

int test_colloid_pointer_array_alloc_managed(void) {

  int ifail = 0;

  /* Host assignment and kernel check */
  {
    int managed = 1;
    int ntotal  = 32;

    colloid_pointer_array_t array = {};

    /* Test data */
    colloid_t * data = NULL;

    tdpAssert( tdpMallocManaged((void **) &data, ntotal*sizeof(colloid_t),
				tdpMemAttachGlobal) );
    for (int n = 0; n < ntotal; n++) {
      data[n] = (colloid_t) {};
      data[n].s.index = 1 + n;
    }

    /* Array and initialise pointer elements ... */
    ifail = colloid_pointer_array_alloc(managed, ntotal, &array);
    assert(ifail == 0);

    for (int n = 0; n < ntotal; n++) {
      array.colloid[n] = data + n;
    }

    /* Check */
    {
      dim3 blocks  = {1, 1, 1};
      dim3 threads = {128, 1, 1};

      blocks.x = ntotal;

      tdpLaunchKernel(kernel3, blocks, threads, 0, 0, ntotal, array);
      tdpAssert( tdpStreamSynchronize(0) );
    }

    /* Finish */
    colloid_pointer_array_free(&array);
    tdpAssert( tdpFree(data) );
  }

  /* Kernel assignment; host check */
  {
    int managed = 1;
    int ntotal  = 32;

    colloid_pointer_array_t array = {};

    /* Attach the test data storage to the array ... */
    colloid_t * data = NULL;

    tdpAssert( tdpMallocManaged((void **) &data, ntotal*sizeof(colloid_t),
				tdpMemAttachGlobal) );

    ifail = colloid_pointer_array_alloc(managed, ntotal, &array);
    assert(ifail == 0);

    for (int n = 0; n < ntotal; n++) {
      array.colloid[n] = data + n;
    }

    /* Assign (kernel) */
    {
      dim3 blocks  = {ntotal, 1, 1};
      dim3 threads = {128, 1, 1};

      tdpLaunchKernel(kernel4, blocks, threads, 0, 0, ntotal, array);
      tdpAssert( tdpStreamSynchronize(0) );
    }

    /* Check (host) */
    for (int n = 0; n < ntotal; n++) {
      if (array.colloid[n]->s.index != 1 + n) printf("Fail at %2d\n", n);
      assert(array.colloid[n]->s.index == 1 + n);
    }

    /* Finish */
    colloid_pointer_array_free(&array);
    tdpAssert( tdpFree(data) );
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_pointer_array_realloc_host
 *
 *****************************************************************************/

int test_colloid_pointer_array_realloc_host(void) {

  int ifail = 0;

  /* Check realloc from nothing */
  {
    int ntotal  = 32;

    colloid_pointer_array_t array = {};

    ifail = colloid_pointer_array_realloc(ntotal, &array);
    assert(ifail == 0);

    assert(array.managed == 0);
    assert(array.ntotal  == ntotal);
    assert(array.colloid);

    for (int n = 0; n < ntotal; n++) {
      array.colloid[n] = NULL;
    }

    colloid_pointer_array_free(&array);
  }

  /* Check realloc preserves existing data */
  {
    int managed  = 0;
    int ntotal   = 32;
    int newtotal = 64;

    colloid_pointer_array_t array = {};
    colloid_t * data = NULL;

    data = (colloid_t *) malloc(ntotal*sizeof(colloid_t));
    assert(data);

    ifail = colloid_pointer_array_alloc(managed, ntotal, &array);
    assert(ifail == 0);

    for (int n = 0; n < ntotal; n++) {
      array.colloid[n] = data + n;
    }

    /* Reallocate */

    ifail = colloid_pointer_array_realloc(newtotal, &array);
    assert(array.managed == managed);
    assert(array.ntotal  == newtotal);
    assert(array.colloid);

    /* Check existing are unchanged and access new */

    for (int n = 0; n < ntotal; n++) {
      assert(array.colloid[n] == data + n);
      array.colloid[ntotal + n] = NULL;
    }

    /* Finish */
    colloid_pointer_array_free(&array);
    free(data);
  }

  return ifail;
}


/*****************************************************************************
 *
 *  test_colloid_pointer_array_realloc_managed
 *
 *****************************************************************************/

int test_colloid_pointer_array_realloc_managed(void) {

  int ifail = 0;

  /* Allocate and set some values using kernel4; re-allocate,
   * set additional values on host, and check with kernel3 */

  {
    int managed  = 1;
    int ntotal   = 32;
    int newtotal = 64;

    colloid_pointer_array_t array = {};

    /* Test data (managed) */
    colloid_t * data = NULL;

    tdpAssert( tdpMallocManaged((void **) &data, newtotal*sizeof(colloid_t),
				tdpMemAttachGlobal) );

    ifail = colloid_pointer_array_alloc(managed, ntotal, &array);
    assert(ifail == 0);

    for (int n = 0; n < ntotal; n++) {
      array.colloid[n] = data + n;
    }

    /* Assign (kernel4) */
    {
      dim3 blocks  = {ntotal, 1, 1};
      dim3 threads = {128, 1, 1};

      tdpLaunchKernel(kernel4, blocks, threads, 0, 0, ntotal, array);
      tdpAssert( tdpStreamSynchronize(0) );
    }

    /* re-allocate and update */
    ifail = colloid_pointer_array_realloc(newtotal, &array);
    assert(array.managed == managed);
    assert(array.ntotal  == newtotal);

    for (int n = ntotal; n < newtotal; n++) {
      data[n].s.index = 1 + n;
      array.colloid[n] = data + n;
    }

    /* Check (kernel3) */
    {
      dim3 blocks  = {newtotal, 1, 1};
      dim3 threads = {128, 1, 1};
      tdpLaunchKernel(kernel3, blocks, threads, 0, 0, newtotal, array);
      tdpAssert( tdpStreamSynchronize(0) );
    }

    /* Finish */
    colloid_pointer_array_free(&array);
    tdpAssert( tdpFree(data) );
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_pointer_array_free
 *
 *****************************************************************************/

int test_colloid_pointer_array_free(void) {

  int ifail = 0;

  /* host */
  {
    int managed = 0;
    int ntotal  = 128;

    colloid_pointer_array_t array = {};

    ifail = colloid_pointer_array_alloc(managed, ntotal, &array);
    assert(array.colloid);

    colloid_pointer_array_free(&array);
    assert(array.managed == 0);
    assert(array.ntotal  == 0);
    assert(array.colloid == NULL);
  }

  /* managed */
  {
    int managed = 1;
    int ntotal  = 256;

    colloid_pointer_array_t array = {};

    ifail = colloid_pointer_array_alloc(managed, ntotal, &array);
    assert(array.colloid);

    colloid_pointer_array_free(&array);
    assert(array.managed == 0);
    assert(array.ntotal  == 0);
    assert(array.colloid == NULL);
  }

  return ifail;
}
