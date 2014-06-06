/*****************************************************************************
 *
 *  gradient_gpu.h
 *
 *  Alan Gray
 *
 *****************************************************************************/

#ifndef _GRADIENT_GPU_H
#define _GRADIENT_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

int phi_gradients_compute_gpu(void);
void set_gradient_option_gpu(char option);
int gradient_gpu_init_h(void);

#ifdef __cplusplus
}
#endif

#endif
