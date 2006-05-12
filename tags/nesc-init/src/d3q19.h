/*****************************************************************************
 *
 *  d3q19.h
 *
 *  D3Q19 definitions.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _D3Q19_H_
#define _D3Q19_H_

enum {NVEL = 19};
enum {LE_N_VEL_XING = 5};

extern const int cv[NVEL][3];
extern const double wv[NVEL];

void d3q19_propagate_binary(void);

#endif
