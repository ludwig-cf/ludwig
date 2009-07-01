/*
 * This header file allows for practically all-platform (ibm, amd, intel, gotoblas, atlas) independent dgemm implementation
 */


#ifdef X86

extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int
		  *ldc, int transa_len, int transb_len);
extern void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx,
		  double *beta, double *y, int *incy, int trans_len);

#define DGEMM(A,B,C,D,E,F,G,H,I,J,K,L,M) dgemm_(A,B,C,D,E,F,G,H,&(I),J,K,&(L),M,H,J)
#define DGEMV(A,B,C,D,E,F,G,H,I,J,K) dgemv_(A,B,C,D,E,F,&(G),H,I,&(J),K,*F) 
#else
#ifdef POWER_ESSL

#include <essl.h>
#define DGEMM(A,B,C,D,E,F,G,H,I,J,K,L,M) dgemm(A,B,*C,*D,*E,*F,G,*H,&(I),*J,*K,&(L),*M)
#define DGEMV(A,B,C,D,E,F,G,H,I,J,K) dgemv(A,*B,*C,*D,E,*F,&(G),*H,*I,&(J),*K)
#else
#ifdef POWER_GOTO

extern void dgemm(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int
		   *ldc, int transa_len, int transb_len);
extern void dgemv(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, 
		   double *beta, double *y, int *incy, int trans_len);

#define DGEMM(A,B,C,D,E,F,G,H,I,J,K,L,M) dgemm(A,B,C,D,E,F,G,H, &(I),J,K,&(L),M,*H,*J)
#define DGEMV(A,B,C,D,E,F,G,H,I,J,K) dgemv(A,B,C,D,E,F,&(G),H,I,&(J),K,*F)
#endif
#endif
#endif

