/*****************************************************************************
 *
 *  decomp.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2012)
 *  Contributing Authors:
 *    Ruairi Short (Rshort@sms.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef DECOMP_H
#define DECOMP_H

void decomp_init();
void decomp_cart_to_pencil(double *in_array, double *out_array);
void decomp_pencil_to_cart(double *in_array, double *out_array);
int index_3d_f (int x, int y, int z, int size[]);
int index_3d_c (int x, int y, int z, int size[]);
void decomp_pencil_sizes(int size[3], int ip);
void decomp_pencil_starts(int start[3], int ip);
int decomp_fftarr_size();
void decomp_finish();

#endif
