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

void COLL_set_Q(void);
void COLL_set_Q_2(void);
void COLL_randomize_Q(double delta_r);
void colloids_fix_swd(void);
void colloids_q_tensor_anchoring_set(const int type);
void colloids_q_tensor_w_set(double w);
void colloids_q_boundary(const double n[3], double qs[3][3], double q0[3][3]);
void colloids_q_boundary_normal(const int index, const int di[3],
				double dn[3]);
double colloids_q_tensor_w(void);

extern struct io_info_t * io_info_scalar_q_;
void scalar_q_io_info_set(struct io_info_t * info);

void jacobi(double (*a)[3], double d[], double (*v)[3], int *nrot);

#endif
