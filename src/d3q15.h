/*****************************************************************************
 *
 *  d3q15.h
 *
 *  D3Q15 definitions.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _D3Q15_H_
#define _D3Q15_H_

enum {NVEL = 15};
enum {LE_N_VEL_XING = 5};

extern const    int cv[NVEL][3];
extern const double wv[NVEL];
extern const    int BC_Map[NVEL];

typedef struct { 
  double f[NVEL], g[NVEL];
} Site;
extern Site * site;

void propagation(void);
void init_ghosts(const double);
void get_ghosts(double []);

#endif
