/*****************************************************************************
 *
 *  colloids_q_tensor.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Juho Lintuvuori (jlintuvu@ph.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COLLOIDS_Q_TENSOR_H
#define COLLOIDS_Q_TENSOR_H

#include "hydro.h"
#include "map.h"
#include "blue_phase.h"

enum colloid_anchoring {ANCHORING_PLANAR, ANCHORING_NORMAL, ANCHORING_FIXED};

#include "colloids.h"

__targetHost__ int colloids_q_cinfo_set(colloids_info_t * cinfo);
__targetHost__ colloids_info_t* colloids_q_cinfo();
__targetHost__ int colloids_fix_swd(colloids_info_t * cinfo, hydro_t * hydro, map_t * map);
__targetHost__ int colloids_q_boundary(const double n[3], double qs[3][3], double q0[3][3],
			int map_status);

__targetHost__ void colloids_q_tensor_anchoring_set(const int type);
__targetHost__ int  colloids_q_tensor_anchoring(void);

__targetHost__ void colloids_q_boundary_normal(const int index, const int di[3],
				double dn[3]);
__target__ int q_boundary_constants(int ic, int jc, int kc, double qs[3][3],
			 const int di[3], int status, double c[3][3],
				    bluePhaseKernelConstants_t* pbpc,
				    colloids_info_t* cinfo);

__targetHost__ void wall_anchoring_set(const int type);

__targetHost__ int blue_phase_wall_w12(double * w1, double * w2);
__targetHost__ int blue_phase_coll_w12(double * w1, double * w2);
__targetHost__ int blue_phase_wall_w12_set(double w1, double w2);
__targetHost__ int blue_phase_coll_w12_set(double w1, double w2);
__targetHost__ int blue_phase_fs(const double dn[3], double qs[3][3], char status,
		  double *fe);

#endif
