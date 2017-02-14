/****************************************************************************
 *
 *  noise_phi_sk.c
 *
 *  This program requires FFTW3 for the Fourier transforms:
 *  details will vary locally.
 *
 *  Quadrature proceeds by binning the structure factor in k-space
 *  on a regular interval.
 *
 *  It should be compiled with something like
 *  
 *  $(CC) -I $(FFTW3)/include length.c -L$(FFTW3)/lib -lfftw3 -lm
 *
 *  where CC is the C compiler and FFTW3 is the location of the
 *  FFTW3 library
 *
 *  A single file, name given on the command line, is assumed to
 *  provide the order parameter data. See the routine read_phi()
 *  below for details.
 *
 *  ./a.out phi-file
 *
 *  Note that all the floating point data are of type float
 *  (assumed to be 4 bytes).
 *
 *  $Id: length_from_sk.c,v 1.1 2009-06-22 09:56:09 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *  (c) T|he University of Edinburgh (2008)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h> 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>

#define NX 64  /* Original data size may be cropped to remove */
#define NY 64  /* artefacts from periodic boundaries */

#define NBIN 32     /* Number of bins for integration in |k| */

double phi[NX][NY];

double sk[NBIN];
double nk[NBIN];

void read_phi(char * filename);

/****************************************************************************
 *
 *  main
 *
 ****************************************************************************/

int main(int argc, char ** argv) {

  int    index, i, j, k, n;
  int    indexdash, idash, jdash;

  /* kx, ky, range from 0-> pi, with the resolution in Fourier
   * space being dk = 2pi/L (assuming system is cubic) */

  double  kmax = 4.0*atan(1.0);
  double  dk = 2.0*kmax/NX;
  double  kx, ky, kmod;
  double  sq;

  fftw_plan      plan;
  fftw_complex  *in, *out;

  /* Set file name and read data */

  read_phi(argv[1]);

  /* fourier transform a subset */

  in  = fftw_malloc(sizeof(fftw_complex)*NX*NY);
  out = fftw_malloc(sizeof(fftw_complex)*NX*NY);

  for (i = 0; i < NX; i++) {
    for (j = 0; j < NY; j++) {
      index = j + NX*i;
      in[index][0] = phi[i][j];
      in[index][1] = 0.0;
    }
  }

  plan = fftw_plan_dft_2d(NX, NY, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(plan);

  /* Compute the structure factor phi(k) phi(-k) and bin according to
   * |k|. */

  for (n = 0; n < NBIN; n++) {
    sk[n] = 0.0;
    nk[n] = 0.0;
  }

  for (i = 0; i < NX; i++) {
    kx = dk*i;
    if (i > NX/2) kx -= 2.0*kmax;
    
    for (j = 0; j < NY/2; j++) {
      ky = dk*j;
      if (j > NY) ky -= 2.0*kmax;

      kmod = sqrt(kx*kx + ky*ky);
      index = j + NX*i;

      sq = out[index][0]*out[index][0] + out[index][1]*out[index][1];
      n = floor(kmod*NBIN/(sqrt(2.0)*kmax));
      assert(n < NBIN);
      sk[n] += sq;
      nk[n] += 1.0;
    }
  }

  /* Quadrature for binned s(k) wrt |k| */

  dk = sqrt(2.0)*kmax/NBIN;

  for (k = 0; k < NBIN; k++) {
    sq = 0.0;
    if (nk[k] != 0) sq = sk[k]/nk[k];
    printf("%3d %5.0f %14.7e % 14.7e\n", k, nk[k], k*dk, sq);
  }

  fftw_free(in);
  fftw_free(out);
  fftw_destroy_plan(plan);

  return 0;
}

/*****************************************************************************
 *
 *  read_phi
 *
 *  It is assumed that the data are stored in a binary file
 *  arranged regularly with the k (y) index running fastest.
 *
 *****************************************************************************/

void read_phi(char * filename) {

  int i, j, n;
  double tmp;

  FILE * fp;

  if( (fp = fopen(filename,"r")) == NULL) {
    fprintf(stderr, "Failed to open phi file %s\n", filename);
    exit(0);
  }

  for (i = 0; i < NX; i++) {
    for (j = 0; j < NY; j++) {
	n = fread(&tmp, sizeof(double), 1, fp);
	phi[i][j] = tmp;
    }
  }

  fclose(fp);

  return;
}


