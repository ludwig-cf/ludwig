/*****************************************************************************
 *
 *  colloid_array_util.h
 *
 *  (c) 2025-2026 The University of Edinburgh
 *
 *  Edinburgh Soft MAtter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *****************************************************************************/

#ifndef LUDWIG_COLLOID_ARRAY_UTIL_H
#define LUDWIG_COLLOID_ARRAY_UTIL_H

#include "colloid.h"

typedef struct colloid_array_s {
  int               managed; /* true if managed memory */
  int               ntotal;  /* number of ... */
  colloid_state_t * data;    /* data items    */
} colloid_array_t;

typedef struct colloid colloid_t;

typedef struct colloid_pointer_array_s {
  int          managed; /* true if managed memory */
  int          nsz;     /* extent of ... */
  colloid_t ** colloid; /* array with pointer entries */
} colloid_pointer_array_t;

int colloid_array_alloc(int managed, int ntotal, colloid_array_t * ptr);
int colloid_array_realloc(int newtotal, colloid_array_t * ptr);
void colloid_array_free(colloid_array_t * ptr);

int colloid_pointer_array_alloc(int managed, int nsz,
                                colloid_pointer_array_t * ptr);
int colloid_pointer_array_realloc(int newsz, colloid_pointer_array_t * ptr);
void colloid_pointer_array_free(colloid_pointer_array_t * pre);

#endif
