#ifndef _COLLOIDS_Q_TENSOR_H
#define _COLLOIDS_Q_TENSOR_H

void COLL_set_Q(void);
void COLL_set_Q_2(void);
void COLL_randomize_Q(double delta_r);
void colloids_fix_swd(void);

void scalar_q_io_init(void);
extern struct io_info_t * io_info_scalar_q_;

void jacobi(double (*a)[3], double d[], double (*v)[3], int *nrot);

#endif
