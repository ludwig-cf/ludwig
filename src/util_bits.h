/*****************************************************************************
 *
 *  util_bits.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022-2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_UTIL_BITS_H
#define LUDWIG_UTIL_BITS_H

#include <stddef.h>

int util_bits_same(size_t size, const void * data1, const void * data2);
int util_double_same(double d1, double d2);
int util_printf_bits(size_t size, const void * data);
int util_set_bit(int mask, int bit);
int util_unset_bit(int mask, int bit);

#endif
