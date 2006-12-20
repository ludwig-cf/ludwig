/*****************************************************************************
 *
 *  d3q19.h
 *
 *  D3Q19 definitions.
 *
 *  $Id: d3q19.h,v 1.5 2006-12-20 16:51:25 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifdef _D3Q19_

enum {NVEL = 19};
enum {LE_N_VEL_XING = 5};

extern const    int cv[NVEL][3];
extern const double wv[NVEL];
extern const double q_[NVEL][3][3];
extern const double norm_[NVEL];
extern const double ma_[NVEL][NVEL];
extern const double mi_[NVEL][NVEL];

void init_ghosts(const double);
void get_ghosts(double []);

#endif
