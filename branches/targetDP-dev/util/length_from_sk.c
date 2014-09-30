/****************************************************************************
 *
 *  length_from_sk.c
 *
 *  This program computes a domain length scale L from the spherically
 *  averaged structure factor S(k). The latter is, in turn, the
 *  Fourier transform of the order parameter distribution phi(r).
 *
 *  This is intended for use with binary fluid calculations
 *  on a cubic lattice size NXxNYxNZ, which may be reduced
 *  to size NXRxNYRxNZR for the purposes of the Fourier
 *  tranform.
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
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>

#define NX 128  /* Original data size may be cropped to remove */
#define NY 128  /* artefacts from periodic boundaries */
#define NZ 128

#define NXR 128  /* Size of cropped data for Fourier transform */
#define NYR 128
#define NZR 128

#define NBIN 128     /* Number of bins for integration in |k| */

#define DUMMY -100.0 /* Dummy phi value to indicate solid points */

float phi[NX][NY][NZ];

float mk[NBIN];
float sk[NBIN];
float nk[NBIN];

void read_phi(char * filename);

/****************************************************************************
 *
 *  main
 *
 *  Computes the length scale L
 *  (see, e.g., Kendon et al. JFM 440 pp147-203 (2001).
 *
 *  If you want the structure factor itself, arrange output
 *  for S(k) below. A dynamic structure factor for given k
 *  can also be arranged if you have a series of files of
 *  order parameter data.
 *
 ****************************************************************************/

int main(int argc, char ** argv) {

  int    index, i, j, k, n;
  int    indexdash, idash, jdash, kdash;

  /* kx, ky, kz, range from 0-> pi, with the resolution in Fourier
   * space being dk = 2pi/L (assuming system is cubic) */

  float  kmax = 4.0*atan(1.0);
  float  dk = 2.0*kmax/NXR;
  float  kx, ky, kz, kmod;
  float  sum1, sum2;
  float  s;

  fftw_plan      plan;
  fftw_complex  *in, *out;

  /* Set file name and read data */

  read_phi(argv[1]);

  /* fourier transform a subset */

  in  = fftw_malloc(sizeof(fftw_complex) * NXR*NYR*NZR);
  out = fftw_malloc(sizeof(fftw_complex) * NXR*NYR*NZR);

  for (i = 0; i < NXR; i++) {
    for (j = 0; j < NYR; j++) {
      for (k = 0; k < NZR; k++) {
	index = k + NYR*(j + NXR*i);
	in[index][0] = phi[i][j][k];
	in[index][1] = 0.0;
      }
    }
  }

  plan = fftw_plan_dft_3d(NXR, NYR, NZR, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

  fftw_execute(plan);

  /* Compute the structure factor phi(k) phi(-k) and bin according to
   * |k|. */

  for (n = 0; n < NBIN; n++) {
    sk[n] = 0.0;
    nk[n] = 0.0;
  }

  for (i = 0; i < NXR; i++) {
    kx = dk*i;
    if (i > NXR/2) kx -= kmax;
    for (j = 0; j < NYR; j++) {
      ky = dk*j;
      if (j > NYR/2) ky -= kmax;
      for (k = 0; k <= NZR/2; k++) {
	kz = dk*k;
	if (k > NYR/2) ky -= kmax;

	kmod = sqrt(kx*kx + ky*ky + kz*kz);

	index = k + NYR*(j + NXR*i);
	idash = NXR - i;
	jdash = NYR - j;
	kdash = NZR - k;
	if (i == 0) idash = 0;
	if (j == 0) jdash = 0;
	if (k == 0) kdash = 0;
	indexdash = kdash + NYR*(jdash + NXR*idash);

	s = out[index][0]*out[index][0] + out[index][1]*out[index][1];
	n = floor(kmod*NBIN/(sqrt(3.0)*kmax));
	sk[n] += s;
	nk[n] += 1.0;
      }
    }
  }

  /* Quadrature for binned s(k) wrt |k| */

  sum1 = 0.0;
  sum2 = 0.0;
  dk = sqrt(3.0)*kmax/NBIN;

  for (k = 0; k < NBIN; k++) {
    kmod = 0.0;
    if (nk[k] != 0) kmod = sk[k]/nk[k];
    kx = (k+0.5)*dk; /* k */
    sum1 += dk*kmod;
    sum2 += kx*dk*kmod;
  }

  /* Work out L */
  kmod = 2.0*4.0*atan(1.0);
  printf("L = %s %g\n", argv[1], kmod*sum1/(sum2));

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
 *  arranged regularly with the k (z) index running fastest.
 *
 *  The data type read is float (4 bytes).
 *
 *  Dummy values are treated as zero (interface).
 *
 *****************************************************************************/

void read_phi(char * filename) {

  int   i, j, k, n;
  float tmp;

  FILE * fp;

  if( (fp = fopen(filename,"r")) == NULL) {
    fprintf(stderr, "Failed to open phi file %s\n", filename);
    exit(0);
  }
  for (i = 0; i < NX; i++) {
    for (j = 0; j < NY; j++) {
      for (k = 0; k < NZ; k++) {
	n = fread(&tmp, sizeof(float), 1, fp);
	if (tmp == DUMMY) tmp = 0.0;
	phi[i][j][k] = tmp;
      }
    }
  }

  fclose(fp);

  return;
}


