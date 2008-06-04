/*****************************************************************************
 *
 *  d3q15.h
 *
 *  D3Q15 definitions.
 *
 *  $Id: d3q15.h,v 1.5.6.1 2008-06-04 19:21:11 erlend Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifdef _D3Q15_

enum {NVEL = 15};
enum {LE_N_VEL_XING = 5};

extern const    int cv[NVEL][3];
extern const double wv[NVEL];
extern const double q_[NVEL][3][3];
extern const double norm_[NVEL];
extern const double ma_[NVEL][NVEL];
extern const double mi_[NVEL][NVEL];

enum {xcount_right = 1};
extern MPI_Datatype types[xcount_right];
extern int xblocklens_right[xcount_right];
extern MPI_Aint xdisp_right_send[xcount_right];
extern MPI_Aint xdisp_right_recv[xcount_right]; 

#endif

