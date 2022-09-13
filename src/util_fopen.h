/*****************************************************************************
 *
 *  util_fopen.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_UTIL_FOPEN_H
#define LUDWIG_UTIL_FOPEN_H

#include <stdio.h>

int util_fopen_default_umask(void);
FILE * util_fopen(const char * path, const char * mode);

#endif
