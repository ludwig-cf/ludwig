/*****************************************************************************
 *
 *  colloid_array_util.c
 *
 *  Manage allocation and deallocate of arrays for:
 *    - colloid_state_t      host, managed
 *    - colloid_state_t *    host, managed
 *    - colloid_t
 *    - colloid_t *          host, managed
 *
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
#include <stdlib.h>
#include <string.h>

#include "target.h"
#include "colloid_array_util.h"

colloid_state_t * colloid_state_allocator(int managed, size_t size);
void colloid_state_deallocator(int managed, colloid_state_t * data);

colloid_t ** colloid_pointer_array_allocator(int managed, size_t size);
void colloid_pointer_array_deallocator(int managed, colloid_t ** ptr);

/*****************************************************************************
 *
 *  colloid_array_alloc
 *
 *****************************************************************************/

int colloid_array_alloc(int managed, int ntotal, colloid_array_t * ptr) {

  int ifail = 0;

  assert(ptr);

  if (ptr == NULL || ntotal <= 0) {
    ifail = -1;
  }
  else {
    ptr->managed = managed;
    ptr->ntotal  = ntotal;
    ptr->data    = colloid_state_allocator(managed, ntotal);
    if (ptr->data == NULL) ifail = -2;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_array_realloc
 *
 *****************************************************************************/

int colloid_array_realloc(int newtotal, colloid_array_t * ptr) {

  int ifail = 0;

  assert(ptr);

  if (newtotal <= 0 || ptr == NULL) {
    ifail = -1;
  }
  else {
    if (ptr->data == NULL) {
      /* Just allocate. We assme managed is set. */
      ifail = colloid_array_alloc(ptr->managed, newtotal, ptr);
    }
    else {
      /* Re-allocate, copy, free ... */
      colloid_state_t * tmp = colloid_state_allocator(ptr->managed, newtotal);
      if (tmp) {
        /* Allow that newtotal is smaller ... */
        int ncopy = (newtotal < ptr->ntotal) ? newtotal : ptr->ntotal;
        memcpy(tmp, ptr->data, ncopy*sizeof(colloid_state_t));
        colloid_state_deallocator(ptr->managed, ptr->data);
        ptr->ntotal = newtotal;
        ptr->data   = tmp;
      }
      ifail = (tmp) ? 0 : 1;
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_array_free
 *
 *  In the spirit of free(), one is allowed to have data = NULL.
 *
 *****************************************************************************/

void colloid_array_free(colloid_array_t * ptr) {

  assert(ptr);

  if (ptr) {
    colloid_state_deallocator(ptr->managed, ptr->data);
    *ptr = (colloid_array_t) {};
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_state_allocator
 *
 *  Utility to perform the actual allocation.
 *
 *****************************************************************************/

colloid_state_t * colloid_state_allocator(int managed, size_t size) {

  colloid_state_t * ptr = NULL;

  assert(size > 0);

  if (managed) {
    tdpAssert(tdpMallocManaged((void **) &ptr, size * sizeof(colloid_state_t),
                               tdpMemAttachGlobal));
  }
  else {
    ptr = (colloid_state_t *) malloc(size * sizeof(colloid_state_t));
  }

  return ptr;
}

/*****************************************************************************
 *
 *  colloid_state_deallocator
 *
 *****************************************************************************/

void colloid_state_deallocator(int managed, colloid_state_t * ptr) {

  if (managed) {
    tdpAssert(tdpFree(ptr));
  }
  else {
    free(ptr);
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_pointer_array_alloc
 *
 *****************************************************************************/

int colloid_pointer_array_alloc(int managed, int nsz,
				colloid_pointer_array_t * ptr) {
  int ifail = 0;

  assert(ptr);

  if (ptr == NULL || nsz <= 0) {
    ifail = -1;
  }
  else {
    ptr->managed = managed;
    ptr->nsz     = nsz;
    ptr->colloid = colloid_pointer_array_allocator(managed, nsz);
    if (ptr->colloid == NULL) ifail = -2;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_pointer_array_realloc
 *
 *****************************************************************************/

int colloid_pointer_array_realloc(int newsz, colloid_pointer_array_t * ptr) {

  int ifail = 0;

  assert(ptr);

  if (newsz <= 0 || ptr == NULL) {
    ifail = -1;
  }
  else {
    if (ptr->colloid == NULL) {
      /* Just allocate. We assme managed is set. */
      ifail = colloid_pointer_array_alloc(ptr->managed, newsz, ptr);
    }
    else {
      /* Re-allocate, copy, free ... */
      colloid_t ** tmp = colloid_pointer_array_allocator(ptr->managed, newsz);
      if (tmp) {
        /* Allow that newtotal is smaller ... */
        int ncopy = (newsz < ptr->nsz) ? newsz : ptr->nsz;
        memcpy(tmp, ptr->colloid, ncopy*sizeof(colloid_t *));
        colloid_pointer_array_deallocator(ptr->managed, ptr->colloid);
        ptr->nsz = newsz;
        ptr->colloid = tmp;
      }
      ifail = (tmp) ? 0 : 1;
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_pointer_array_free
 *
 *****************************************************************************/

void colloid_pointer_array_free(colloid_pointer_array_t * ptr) {

  if (ptr) {
    if (ptr->colloid) {
      colloid_pointer_array_deallocator(ptr->managed, ptr->colloid);
    }
    *ptr = (colloid_pointer_array_t) {};
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_pointer_array_allocator
 *
 *  And initialise entries to NULL.
 *
 *****************************************************************************/

colloid_t ** colloid_pointer_array_allocator(int managed, size_t size) {

  colloid_t ** ptr = NULL;

  assert(size > 0);

  if (managed) {
    tdpAssert(tdpMallocManaged((void **) &ptr, size * sizeof(colloid_t *),
                               tdpMemAttachGlobal));
    tdpAssert( tdpMemset(ptr, 0, size*sizeof(colloid_t *)) );
  }
  else {
    ptr = (colloid_t **) calloc(size, sizeof(colloid_t *));
  }

  return ptr;
}

/*****************************************************************************
 *
 *  colloid_pointer_array_deallocator
 *
 *****************************************************************************/

void colloid_pointer_array_deallocator(int managed, colloid_t ** ptr) {

  assert(ptr);

  if (managed) {
    tdpAssert( tdpFree(ptr) );
  }
  else {
    free(ptr);
  }

  return;
}
