/*
! This file is part of P3DFFT library
!
! Version 2.4
!
! Copyright (C) 2006-2010 Dmitry Pekurovsky
! Copyright (C) 2006-2010 UC San Diego
!
! P3DFFT is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
! P3DFFT is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!
! GNU General Public License for more details.
! You should have received a copy of the GNU General Public License
! along with P3DFFT.  If not, see <http://www.gnu.org/licenses/>.
!
*/

#include <stdlib.h>

#ifdef IBM

#define FORT_MOD_NAME(NAME) __p3dfft_NMOD_##NAME
#define FORTNAME(NAME) NAME

#elif defined INTEL

#define FORT_MOD_NAME(NAME) p3dfft_mp_##NAME##_
#define FORTNAME(NAME) NAME##_

#elif defined PGI

#define FORT_MOD_NAME(NAME) p3dfft_##NAME##_
#define FORTNAME(NAME) NAME##_

#elif defined GNU
#include "gnu.h"

#else

#define FORT_MOD_NAME(NAME) p3dfft_mp_##NAME##_
#define FORTNAME(NAME) NAME##_

#endif

extern void FORT_MOD_NAME(p3dfft_setup)(int *dims,int *nx,int *ny,int *nz, int *ow, int *memsize);
extern void FORT_MOD_NAME(p3dfft_get_dims)(int *,int *,int *,int *);
extern void FORT_MOD_NAME(get_timers)(double *timers);
extern void FORT_MOD_NAME(set_timers)();

#ifndef SINGLE_PREC
extern void FORT_MOD_NAME(p3dfft_ftran_r2c)(double *A,double *B, unsigned char *op);
extern void FORT_MOD_NAME(p3dfft_btran_c2r)(double *A,double *B, unsigned char *op);
#else
extern void FORT_MOD_NAME(p3dfft_ftran_r2c)(float *A,float *B, unsigned char *op);
extern void FORT_MOD_NAME(p3dfft_btran_c2r)(float *A,float *B, unsigned char *op);
#endif

extern void FORT_MOD_NAME(p3dfft_clean)();

extern void p3dfft_setup(int *dims,int nx,int ny,int nz,int ovewrite, int *memsize);
extern void p3dfft_get_dims(int *,int *,int *,int );

#ifndef SINGLE_PREC
extern void p3dfft_ftran_r2c(double *A,double *B, unsigned char *op);
extern void p3dfft_btran_c2r(double *A,double *B, unsigned char *op);
#else
extern void p3dfft_ftran_r2c(float *A,float *B, unsigned char *op);
extern void p3dfft_btran_c2r(float *A,float *B, unsigned char *op);
#endif

extern void p3dfft_clean();

extern void get_timers(double *timers);
extern void set_timers();

inline void get_timers(double *timers) {
  FORT_MOD_NAME(get_timers)(timers);
}

inline void set_timers() {
  FORT_MOD_NAME(set_timers)();
}

inline void p3dfft_setup(int *dims,int nx,int ny,int nz, int overwrite, int * memsize)
{
  FORT_MOD_NAME(p3dfft_setup)(dims,&nx,&ny,&nz,&overwrite,memsize);
}

inline void p3dfft_get_dims(int *start,int *end,int *size,int conf)
{
  FORT_MOD_NAME(p3dfft_get_dims)(start,end,size,&conf);
}

#ifndef SINGLE_PREC
inline void p3dfft_ftran_r2c(double *A,double *B, unsigned char *op)
{
  FORT_MOD_NAME(p3dfft_ftran_r2c)(A,B,op);
}
#else
inline void p3dfft_ftran_r2c(float *A,float *B, unsigned char *op)
{
  FORT_MOD_NAME(p3dfft_ftran_r2c)(A,B,op);
}

#endif

#ifndef SINGLE_PREC
inline void p3dfft_btran_c2r(double *A,double *B, unsigned char *op)
{
  FORT_MOD_NAME(p3dfft_btran_c2r)(A,B,op);
}
#else
inline void p3dfft_btran_c2r(float *A,float *B, unsigned char *op)
{
  FORT_MOD_NAME(p3dfft_btran_c2r)(A,B,op);
}
#endif

inline void p3dfft_clean()		
{		
  FORT_MOD_NAME(p3dfft_clean)();		
}
