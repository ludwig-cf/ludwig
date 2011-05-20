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

enum colloid_anchoring {ANCHORING_PLANAR, ANCHORING_NORMAL, ANCHORING_FIXED};
enum colloid_anchoring_method {ANCHORING_METHOD_NONE, ANCHORING_METHOD_ONE,
			       ANCHORING_METHOD_TWO};

void COLL_set_Q(void);
void colloids_fix_swd(void);
void colloids_q_tensor_anchoring_set(const int type);
void colloids_q_tensor_w_set(double w);
void colloids_q_boundary(const double n[3], double qs[3][3], double q0[3][3],
			 char site_map_status);
void colloids_q_boundary_normal(const int index, const int di[3],
				double dn[3]);
double colloids_q_tensor_w(void);
void colloids_q_anchoring_method_set(int method);
int  colloids_q_anchoring_method(void);
void wall_anchoring_set(const int type);

extern struct io_info_t * io_info_scalar_q_;
void scalar_q_io_info_set(struct io_info_t * info);

#endif
