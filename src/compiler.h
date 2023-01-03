/*****************************************************************************
 *
 *  compiler.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_COMPILER_H
#define LUDWIG_COMPILER_H

typedef struct compiler_info_s compiler_info_t;

struct compiler_info_s {
  int cplusplus;          /* __cplusplus */
  int major;              /* Major */
  int minor;              /* Minor */
  int patchlevel;         /* Patch */
  const char * version;   /* Version string */
  const char * name;      /* Vendor major.minor.patch */
  const char * options;   /* Compiler options at compile time ("CFLAGS") */
  const char * commit;    /* Git 40-character commit hash */
};

int compiler_id(compiler_info_t * compiler);

#endif
