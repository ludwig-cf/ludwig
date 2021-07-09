/*****************************************************************************
 *
 *  compiler.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_COMPILER_H
#define LUDWIG_COMPILER_H

#include <stdio.h>

typedef struct compiler_info_s compiler_info_t;

struct compiler_info_s {
  int cplusplus;          /* __cplusplus */
  int major;              /* Major */
  int minor;              /* Minor */
  int patchlevel;         /* Patch */
  char version[BUFSIZ];   /* Version string */
  char name[BUFSIZ];      /* Vendor major.minor.patch */
};

int compiler_id(compiler_info_t * compiler);

#endif
