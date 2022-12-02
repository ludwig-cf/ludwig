/*****************************************************************************
 *
 *  util_bits.c
 *
 *  Various bit/byte manipulations
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Center
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>

/*****************************************************************************
 *
 *  util_bits_same
 *
 *  Return 1 if two bits arrays are the same.
 *
 *****************************************************************************/

int util_bits_same(size_t size, const void * data1, const void * data2) {

  int bsame = 1;
  unsigned char * b1 = (unsigned char *) data1;
  unsigned char * b2 = (unsigned char *) data2;

  assert(b1);
  assert(b2);

  for (size_t ibyte = size; ibyte-- > 0; ) {
    for (int ibit = 7; ibit >= 0; ibit--) {
      unsigned char bit1 = (b1[ibyte] >> ibit) & 1;
      unsigned char bit2 = (b2[ibyte] >> ibit) & 1;
      if (bit1 != bit2) bsame = 0;
    }
  }

  return bsame;
}

/*****************************************************************************
 *
 *  util_double_same
 *
 *  Return 1 if two doubles are bitwise the same; zero if not.
 *
 *****************************************************************************/

int util_double_same(double d1, double d2) {

  double a = d1;
  double b = d2;

  return util_bits_same(sizeof(double), &a, &b);
}

/*****************************************************************************
 *
 *  util_printf_bits
 *
 *  Convenience to print a set of bits as 0 or 1.
 *
 *****************************************************************************/

int util_printf_bits(size_t size, const void * data) {

  unsigned char * bytes = (unsigned char *) data;

  if (data == NULL) return -1;

  for (size_t ibyte = size; ibyte-- > 0; ) {
    for (int ibit = 7; ibit >= 0; ibit--) {
      unsigned char bit = (bytes[ibyte] >> ibit) & 1;
      printf("%u", bit);
    }
  }

  printf("\n");

  return 0;
}
