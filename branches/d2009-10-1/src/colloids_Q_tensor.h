#ifndef _COLLOIDS_Q_TENSOR_H
#define _COLLOIDS_Q_TENSOR_H

void COLL_set_Q(void);
void COLL_randomize_Q(double delta_r);
#define n 3
void jacobi(double (*a)[n], double d[], double (*v)[n], int *nrot);
#undef n
#endif
