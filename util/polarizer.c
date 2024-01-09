/****************************************************************************
 *
 *  polarizer.c
 *
 *  This program simulates the appearance of a liquid-crystalline sample
 *  under polarised light. Director and scalar order parameter data in 
 *  vtk-format is required. The output consists of a file 'polar-...' 
 *  that contains the light intensity in vtk-format.
 * 
 *  The file names should be (in the current directory) e.g.,:
 *
 *  ./polarizer lcd-00100000.vtk lcs-00100000.vtk x|y|z
 * 
 *  Cartesian_direction is x, y, or z and defines the direction of the
 *  incident light.
 *
 *  COMMAND LINE INPUT
 *  (1) the director filename (must be "lcd-*.vtk")
 *  (2) the scalar order parameter filename
 *  (3) the Cartesian direction of incident light
 *
 *  See 
 *  (1) Craig F. Bohren, Donald R. Huffman,  "Absorption and Scattering 
 *  of Light by Small Particles" and
 *  (2) E. Berggren, C. Zannoni, C. Chicolli, P. Pasini, F. Semeria, 
 *  Phys. Rev. E 50, 2929 (1994) for more information on the physical model.
 * 
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Oliver Henrich  (oliver.henrich@strath.ac.uk)
 *  University of Strathclyde, Glasgow, UK
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "util_fopen.h"

#define Pi 3.141592653589793
#define NLAMBDA 3

typedef struct options_s {
  int raydir;               /* Direction of incident ray: 0,1,2 = X,Y,Z */
  int is_nematic;           /* true will set scalar order parameter = 1/3 */
  int cut_topbot;           /* remove cut_topbot sites at 'entry' and 'exit' */
  double xi_polarizer;      /* polariser angle (degrees) */
  double xi_analyzer;       /* analyser angle (degrees) */
  double n_e;               /* extraordinary index of refraction */
  double n_o;               /* ordinary index of refraction */
  double lambda[NLAMBDA];   /* wavelengths (lattice units) */
  double weight[NLAMBDA];   /* weights (sum should be unity) */
} options_t;

/* Set your options here: */

options_t default_options(void) {

  options_t default_options = {
    .raydir = 0,
    .is_nematic = 0,
    .cut_topbot = 0,
    .xi_polarizer = 0.0,
    .xi_analyzer  = 90.0,
    .n_e = 2.0,
    .n_o = 1.5,
    .lambda = { 18.0, 20.0, 22.0},
    .weight = { 0.2,   0.4,  0.2}
  };

  return default_options;
}

/*** IT SHOULD NOT BE NECESSARY TO MODIFY ANYTHING BELOW THIS LINE ***/

typedef struct system_s {
  int Lx;                 /* System size ... */
  int Ly;
  int Lz;
  int its;                /* time step (from director file name) */
  double ****  dir;       /* Director field */
  double ***   sop;       /* Scalar order parameter field */
  double ***** mueller;   /* Mueller retarder matrix at each site [4][4] */
  double **    s_out;     /* Output intensity 2d pattern (Stokes vector[0]) */
} system_t;

void allocate(const options_t * opts, system_t * sys);
void read_data(int argc, char ** argv, const options_t * opts, system_t * sys);
void initialise_matrices(const options_t * opts, int ilambda, system_t * sys);
void simulate_polarizer(const options_t * opts, int ilambda, system_t * sys);
void output(int argc, char ** argv, const options_t * opts, system_t * sys);
void polariser_matrix(double angle, double p[4][4]);
void print_usage(void);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char* argv[]){

  char filename[BUFSIZ] = {0};
  char line[BUFSIZ] = {0};       /* line of input */
  char dummy[BUFSIZ] = {0};      /* unused input */

  FILE * dirinput = NULL;
  char * pl = NULL;

  options_t opts = default_options();
  system_t sys = {0};
  int nread = 0;

  assert(argc > 1);

  if (opts.is_nematic && argc == 3) {
    if (*argv[2] == 'x') opts.raydir = 0;
    if (*argv[2] == 'y') opts.raydir = 1;
    if (*argv[2] == 'z') opts.raydir = 2;
  }
  else if (!opts.is_nematic && argc == 4) {
    if (*argv[3] == 'x') opts.raydir = 0;
    if (*argv[3] == 'y') opts.raydir = 1;
    if (*argv[3] == 'z') opts.raydir = 2;
  }
  else {
    print_usage();
    exit(-1);
  }

  /* We assume a file name: lcd-01234567.vtk; work out the time step ... */
  {
    char * endptr = NULL; /* should be the first invalid char, ie., '.' */
    int its = strtol(argv[1] + 4, &endptr, 10);

    if (*endptr == '.') sys.its = its;
    assert(0 <= sys.its && sys.its < 1000*1000*1000);
  }

  /* take system dimensions from vtk-header */

  sprintf(filename, "lcd-%8.8d.vtk", sys.its);
  dirinput = util_fopen(filename, "r");

  if (!dirinput) {
    printf("Cannot open director input file %s\n", argv[1]);
    exit(-1);
  }

  for (int skip = 0; skip < 4; skip++) {
    pl = fgets(line, BUFSIZ, dirinput);
    assert(pl);
    if (skip == 1) printf("Director input: %s\n", pl);
  }

  nread = fscanf(dirinput, "%s %d %d %d", dummy, &sys.Lx, &sys.Ly, &sys.Lz);
  assert(nread == 4);
  if (nread != 4) {
    printf("File %s: unexpected format\n", filename);
  }


  for (int skip = 5; skip < 10; skip++) {
    pl = fgets(line, BUFSIZ, dirinput);
    assert(pl);
  }

  fclose(dirinput);

  allocate(&opts, &sys);

  read_data(argc, argv, &opts, &sys);

  /* For each wavelength ... */

  for (int ia = 0; ia < NLAMBDA; ia++) {

    printf("# Wavelength no %d: lambda=%g weight=%g\n", ia+1,
	   opts.lambda[ia], opts.weight[ia]);

    initialise_matrices(&opts, ia, &sys);
    simulate_polarizer(&opts, ia, &sys);
  }

  output(argc, argv, &opts, &sys);

  printf("# Done\n");

  return 0;
}

/*****************************************************************************
 *
 *  print_usage
 *
 *  Display instructions (usually before bailing out).
 *
 *****************************************************************************/

void print_usage(void) {

  printf("Usage: ./polarizer lcd-00000000.vtk [lcs-00000000.vtk] x|y|z\n");

  return;
}

/*****************************************************************************
 *
 *  polairsier_matrix
 *
 *  Ideal linear polariser for (smalles) angle "angle" between the
 *  transmission axis and the electric field vector (assumed to be
 *  at right angles to coordinate direction).
 *
 *  The input "angle" is in degrees.
 *
 *  Eg., Bohren and Huffman (Eq. 2.87)
 *
 *****************************************************************************/

void polariser_matrix(double angle, double p[4][4]) {

  double xi = (Pi/180.0)*angle;
  double c2xi = cos(2.0*xi);
  double s2xi = sin(2.0*xi);
  const double r2 = 0.5;

  p[0][0] = r2*1.0;
  p[0][1] = r2*c2xi;
  p[0][2] = r2*s2xi;
  p[0][3] = 0.0;

  p[1][0] = r2*c2xi;
  p[1][1] = r2*c2xi*c2xi;
  p[1][2] = r2*c2xi*s2xi;
  p[1][3] = 0.0;

  p[2][0] = r2*s2xi;
  p[2][1] = r2*s2xi*c2xi;
  p[2][2] = r2*s2xi*s2xi;
  p[2][3] = 0.0;

  p[3][0] = 0.0;
  p[3][1] = 0.0;
  p[3][2] = 0.0;
  p[3][3] = 0.0;

  return;
}

/*****************************************************************************
 *
 *  allocate
 *
 *****************************************************************************/

void allocate(const options_t * opts, system_t * sys) {

  int Lx = sys->Lx;
  int Ly = sys->Ly;
  int Lz = sys->Lz;

  sys->dir = (double****) calloc(Lx, sizeof(double ***));
  sys->sop = (double***) calloc(Lx, sizeof(double **));
  sys->mueller = (double*****) calloc(Lx, sizeof(double ****));

  for (int i = 0; i < Lx; i++) {

    sys->dir[i] = (double***) calloc(Ly, sizeof(double **));
    sys->sop[i] = (double**) calloc(Ly, sizeof(double *));
    sys->mueller[i] = (double****) calloc(Ly, sizeof(double ***));

    for (int j = 0; j < Ly; j++){

      sys->dir[i][j] = (double **) calloc(Lz, sizeof(double *));
      sys->sop[i][j] = (double * ) calloc(Lz, sizeof(double));
      sys->mueller[i][j] = (double ***) calloc(Lz, sizeof(double **));

      for (int k = 0; k < Lz; k++){

        sys->dir[i][j][k] = (double *) calloc(3, sizeof(double));
        sys->mueller[i][j][k] = (double**) calloc(4, sizeof(double *));

        for (int m = 0; m < 4; m++){
          sys->mueller[i][j][k][m] = (double*) calloc(4, sizeof(double));
        }
      }
    }
  }

  if (opts->raydir == 0) {

    sys->s_out = (double **) calloc(Ly, sizeof(double *));

    for (int j = 0; j < Ly; j++) {
      sys->s_out[j] = (double *) calloc(Lz, sizeof(double));
    }
  }

  if (opts->raydir == 1) {

    sys->s_out = (double **) calloc(Lx, sizeof(double *));

    for (int i = 0; i < Lx; i++) {
      sys->s_out[i] = (double *) calloc(Lz, sizeof(double));
    }
  }

  if (opts->raydir == 2) {

    sys->s_out = (double **) calloc(Lx, sizeof(double *));

    for (int i = 0; i < Lx; i++) {
      sys->s_out[i] = (double *) calloc(Ly, sizeof(double));
    }
  }

  return;
}

/*****************************************************************************
 *
 *  read_data
 *
 *****************************************************************************/

void read_data(int argc, char** argv, const options_t * opts,
	       system_t * sys) {

  char filename[BUFSIZ] = {0};
  char line[BUFSIZ] = {0};

  printf("# Director input\n");


  if (argc < 2) {
    print_usage();
    exit(-1);
  }
  else {

    FILE * dirinput = NULL;
    char * pl = NULL;

    sprintf(filename, "lcd-%8.8d.vtk", sys->its);
    dirinput = util_fopen(filename, "r");

    if (!dirinput) {
      printf("Cannot open director input file %s\n", argv[1]);
      exit(-1);
    }


    /* skip vtk header lines */
    for (int skip = 0; skip < 9; skip++) {
      pl = fgets(line, BUFSIZ, dirinput);
      if (pl == NULL) printf("Error in header\n");
    }

    for (int k = 0; k < sys->Lz; k++) {
      for (int j = 0; j < sys->Ly; j++) {
	for (int i = 0; i < sys->Lx; i++) {

	  pl = fgets(line, BUFSIZ, dirinput);
	  assert(pl);

	  double dirx = 0.0;
	  double diry = 0.0;
	  double dirz = 0.0;

	  int ns = sscanf(line, "%le %le %le", &dirx, &diry, &dirz);
	  if (ns == 3) {
	    sys->dir[i][j][k][0] = dirx;
	    sys->dir[i][j][k][1] = diry;
	    sys->dir[i][j][k][2] = dirz;
	  }
	  else {
	    printf("Incorrect dirx diry dirz\n");
	  }
	}
      }
    }

    printf("# Director input complete\n");
  }

  printf("# Scalar order parameter input\r");

  if (opts->is_nematic) {

    printf("# Assuming constant scalar order parameter\n");

    for (int i = 0; i < sys->Lx; i++) {
      for (int j = 0; j < sys->Ly; j++) {
	for (int k = 0; k < sys->Lz; k++) {
	  sys->sop[i][j][k] = 0.3333333;
	}
      }
    }
  }
  else {

    FILE * sopinput = NULL;
    char * pl = NULL;
    char dummy[BUFSIZ] = {0};
    int nread = 0;

    int Lxsop = -1;
    int Lysop = -1;
    int Lzsop = -1;

    sprintf(filename, "lcs-%8.8d.vtk", sys->its);
    sopinput = util_fopen(filename, "r");

    if (!sopinput) {
      printf("Cannot open scalar order parameter input file %s\n", argv[2]);
      exit(0);
    }
    /* skip header vtk lines to size ... */
    for (int skip = 0; skip < 4; skip++) {
      pl = fgets(line, BUFSIZ, sopinput);
      if (pl == NULL) printf("Error in vtk header\n");
    }
    /* take system dimensions from vtk-header */
    nread = fscanf(sopinput,"%s %d %d %d", dummy, &Lxsop, &Lysop, &Lzsop);
    assert(nread == 4);
    if (nread != 4) {
      printf("Bad dimensions in header\n");
      exit(-1);
    }

    /* skip rest header lines */
    for (int skip = 5; skip < 11; skip++) {
      pl = fgets(line, BUFSIZ, sopinput);
      assert(pl);
    }

    /* compare dimensions for consistency */
    if (sys->Lx != Lxsop || sys->Ly != Lysop || sys->Lz != Lzsop) {
      printf("Inconsistent dimensions in director and scalar OP input\n");
      exit(0);
    }

    for (int k = 0; k < sys->Lz; k++) {
      for (int j = 0; j < sys->Ly; j++) {
	for (int i = 0; i < sys->Lx; i++) {
	  pl = fgets(line, BUFSIZ, sopinput);
	  assert(pl);
	  sscanf(line,"%le", &sys->sop[i][j][k]);
	}
      }
    }

    printf("# Scalar order parameter input complete\n");
  }

  return;
}

/*****************************************************************************
 *
 *  initialise_matrices
 *
 *  Computes the Mueller matrices for given wavelength.
 *
 *****************************************************************************/

void initialise_matrices(const options_t * opts, int ilambda, system_t * sys) {

  printf("# Initialisation for waelength %2d\r", ilambda);

  for (int i = 0; i < sys->Lx; i++) {
    for (int j = 0; j < sys->Ly; j++) {
      for (int k = 0; k < sys->Lz; k++) {

	double alpha[3] = {0.0, 0.0, 0.0};
	double beta[3]  = {0.0, 0.0, 0.0};
	double delta = 0.0;

	/* angle between ray direction and local director  */

	alpha[0] = acos(sys->dir[i][j][k][0]);
	alpha[1] = acos(sys->dir[i][j][k][1]);
	alpha[2] = acos(sys->dir[i][j][k][2]);

	/* angle between axis and projection onto coordinate plane */

	beta[0] = atan(sys->dir[i][j][k][2]/sys->dir[i][j][k][1]);
	beta[1] = atan(sys->dir[i][j][k][0]/sys->dir[i][j][k][2]);
	beta[2] = atan(sys->dir[i][j][k][1]/sys->dir[i][j][k][0]);

	{
	  double cosa = cos(alpha[opts->raydir]);
	  double re   = opts->n_e;
	  double ro   = opts->n_o;
	  double lambda = opts->lambda[ilambda];
	  double rej    = sqrt(ro*ro + (re*re - ro*ro)*cosa*cosa);

	  delta = 2.0*Pi*sys->sop[i][j][k]*ro*(re/rej - 1.0)/lambda;
	}

	{
	  double sd = sin(delta);
	  double cd = cos(delta);
	  double sb = sin(2.0*beta[opts->raydir]);
	  double cb = cos(2.0*beta[opts->raydir]);

	  sys->mueller[i][j][k][0][0] = 1.0;
	  sys->mueller[i][j][k][0][1] = 0.0;
	  sys->mueller[i][j][k][0][2] = 0.0;
	  sys->mueller[i][j][k][0][3] = 0.0;

	  sys->mueller[i][j][k][1][0] = 0.0;
	  sys->mueller[i][j][k][1][1] = cb*cb + sb*sb*cd;
	  sys->mueller[i][j][k][1][2] = sb*cb*(1.0 - cd);
	  sys->mueller[i][j][k][1][3] = -sb*sd;

	  sys->mueller[i][j][k][2][0] = 0.0;
	  sys->mueller[i][j][k][2][1] = sb*cb*(1.0 - cd);
	  sys->mueller[i][j][k][2][2] = sb*sb + cb*cb*cd;
	  sys->mueller[i][j][k][2][3] = cb*sd;

	  sys->mueller[i][j][k][3][0] = 0.0;
	  sys->mueller[i][j][k][3][1] = sb*sd;
	  sys->mueller[i][j][k][3][2] = -cb*sd;
	  sys->mueller[i][j][k][3][3] = cd;
	}
      }
    }
  }

  printf("# Initialisation complete\n");

  return;
}

/*****************************************************************************
 *
 *  simulate_polarizer
 *
 *  Calculates the intensity for a given light component.
 *
 *****************************************************************************/

void simulate_polarizer(const options_t * opts, int ilambda, system_t * sys) {

  int cut_topbot = opts->cut_topbot;
  double weight0 = opts->weight[ilambda];

  double p1[4][4] = {0};                       /* Polariser */
  double p2[4][4] = {0};                       /* Analyser */

  const double s_in[4] = {1.0, 0.0, 0.0, 0.0}; /* Incident beam */

  printf("# Simulating polarizer\n");

  polariser_matrix(opts->xi_polarizer, p1);
  polariser_matrix(opts->xi_analyzer,  p2);

  if (opts->raydir == 0) {

    for (int j = 0; j < sys->Ly; j++) {
      for (int k = 0; k < sys->Lz; k++) {

	double sp_inp[4] = {0};
	double sp_out[4] = {0};

	/* Input polariser */
        for (int m = 0; m < 4; m++) {
          for (int n = 0; n < 4; n++) {
            sp_inp[m] += p1[m][n]*s_in[n];
          }
        }

	/* Product along the system x-direction. */

        for (int i = 0 + cut_topbot; i < sys->Lx - cut_topbot; i++) {

	  for (int m = 0; m < 4; m++) {
	    for (int n = 0; n < 4; n++) {
	      sp_out[m] += sys->mueller[i][j][k][m][n]*sp_inp[n];
	    }
	  }
	  for (int n = 0; n < 4; n++) {
	    sp_inp[n] = sp_out[n];
	    sp_out[n] = 0.0;
	  }
        }

	/* Output polariser */
        for (int m = 0; m < 4; m++) {
          for (int n = 0; n < 4; n++) {
            sp_out[m] += p2[m][n]*sp_inp[n];
          }
        }
	sys->s_out[j][k] += weight0*sp_out[0];
      }
    }
  }

  if (opts->raydir == 1) {

    for (int i = 0; i < sys->Lx; i++) {
      for (int k = 0; k < sys->Lz; k++) {

	double sp_inp[4] = {0};
	double sp_out[4] = {0};

	/* Input polariser */
        for (int m = 0; m < 4; m++) {
          for (int n = 0; n < 4; n++) {
            sp_inp[m] += p1[m][n]*s_in[n];
          }
        }

	/* Along the system in y-direction .. */

        for (int j = 0 + cut_topbot; j < sys->Ly - cut_topbot; j++) {

          for (int m = 0; m < 4; m++) {
            for (int n = 0; n < 4; n++) {
	      sp_out[m] += sys->mueller[i][j][k][m][n]*sp_inp[n];
	    }
	  }
	  for (int n = 0; n < 4; n++) {
	    sp_inp[n] = sp_out[n];
	    sp_out[n] = 0.0;
	  }
        }

	/* output polsariser */

        for (int m = 0; m < 4; m++) {
          for (int n = 0; n < 4; n++) {
            sp_out[m] += p2[m][n]*sp_inp[n];
          }
        }
	sys->s_out[i][k] += weight0*sp_out[0];
      }
    }
  }

  if (opts->raydir == 2) {

    for (int i = 0; i < sys->Lx; i++) {
      for (int j = 0; j < sys->Ly; j++) {

	double sp_inp[4] = {0};
	double sp_out[4] = {0};

	/* Input polariser */
        for (int m = 0; m < 4; m++) {
          for (int n = 0; n < 4; n++) {
            sp_inp[m] += p1[m][n]*s_in[n];
          }
        }

	/* along the length in z-direction ... */

        for (int k = 0 + cut_topbot; k < sys->Lz - cut_topbot; k++) {

          for (int m = 0; m < 4; m++) {
            for (int n = 0; n < 4; n++) {
	      sp_out[m] += sys->mueller[i][j][k][m][n]*sp_inp[n];
	    }
	  }
	  for (int n = 0; n < 4; n++) {
	    sp_inp[n] = sp_out[n];
	    sp_out[n] = 0.0;
	  }
	}

	/* analyser ... */
        for (int m = 0; m < 4; m++) {
          for (int n = 0; n < 4; n++) {
            sp_out[m] += p2[m][n]*sp_inp[n];
          }
        }
	sys->s_out[i][j] += weight0*sp_out[0];
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  output
 *
 *****************************************************************************/

void output(int argc, char ** argv, const options_t * opts, system_t * sys) {

  FILE * polaroutput = NULL;
  char outfile[BUFSIZ] = {0};

  printf("# Output\n");

  /* Output filename from time step etc */

  if (opts->raydir == 0) sprintf(outfile, "polar-x-%8.8d.vtk", sys->its);
  if (opts->raydir == 1) sprintf(outfile, "polar-y-%8.8d.vtk", sys->its);
  if (opts->raydir == 2) sprintf(outfile, "polar-z-%8.8d.vtk", sys->its);

  polaroutput = util_fopen(outfile, "w");
  assert(polaroutput);

  if (opts->raydir == 0) {

    /* Average y,z */

    int larea = sys->Ly*sys->Lz;
    double average = 0.0;

    fprintf(polaroutput, "# vtk DataFile Version 2.0\n");
    fprintf(polaroutput, "Generated by ludwig extract.c\n");
    fprintf(polaroutput, "ASCII\n");
    fprintf(polaroutput, "DATASET STRUCTURED_POINTS\n");
    fprintf(polaroutput, "DIMENSIONS 1 %d %d\n", sys->Ly, sys->Lz);
    fprintf(polaroutput, "ORIGIN 0 0 0\n");
    fprintf(polaroutput, "SPACING 1 1 1\n");
    fprintf(polaroutput, "POINT_DATA %d\n", larea);
    fprintf(polaroutput, "SCALARS Polarizer float 1\n");
    fprintf(polaroutput, "LOOKUP_TABLE default\n");

    for (int k = 0; k < sys->Lz; k++) {
      for (int j = 0; j < sys->Ly; j++) {
        fprintf(polaroutput, "%le\n", sys->s_out[j][k]);
	average += sys->s_out[j][k];
      }
    }

    printf("# Output complete\n");
    printf("# Average intensity: %g\n", average/(1.0*larea));
  }

  if (opts->raydir == 1) {

    /* Average x,z */

    int larea = sys->Lx*sys->Lz;
    double average = 0.0;

    fprintf(polaroutput, "# vtk DataFile Version 2.0\n");
    fprintf(polaroutput, "Generated by ludwig extract.c\n");
    fprintf(polaroutput, "ASCII\n");
    fprintf(polaroutput, "DATASET STRUCTURED_POINTS\n");
    fprintf(polaroutput, "DIMENSIONS %d 1 %d\n", sys->Lx, sys->Lz);
    fprintf(polaroutput, "ORIGIN 0 0 0\n");
    fprintf(polaroutput, "SPACING 1 1 1\n");
    fprintf(polaroutput, "POINT_DATA %d\n", larea);
    fprintf(polaroutput, "SCALARS Polarizer float 1\n");
    fprintf(polaroutput, "LOOKUP_TABLE default\n");


    for (int k = 0; k < sys->Lz; k++) {
      for (int i = 0; i < sys->Lx; i++) {
        fprintf(polaroutput, "%le\n", sys->s_out[i][k]);
	average += sys->s_out[i][k];
      }
    }

    printf("# Output complete\n");
    printf("# Average intensity: %g\n", average/(1.0*larea));
  }


  if (opts->raydir == 2) {

    /* Average x,y */

    int larea = sys->Lx*sys->Ly;
    double average = 0.0;

    fprintf(polaroutput, "# vtk DataFile Version 2.0\n");
    fprintf(polaroutput, "Generated by ludwig extract.c\n");
    fprintf(polaroutput, "ASCII\n");
    fprintf(polaroutput, "DATASET STRUCTURED_POINTS\n");
    fprintf(polaroutput, "DIMENSIONS %d %d 1\n", sys->Lx, sys->Ly);
    fprintf(polaroutput, "ORIGIN 0 0 0\n");
    fprintf(polaroutput, "SPACING 1 1 1\n");
    fprintf(polaroutput, "POINT_DATA %d\n", larea);
    fprintf(polaroutput, "SCALARS Polarizer float 1\n");
    fprintf(polaroutput, "LOOKUP_TABLE default\n");

    for (int j = 0; j < sys->Ly; j++) {
      for (int i = 0; i < sys->Lx; i++) {
        fprintf(polaroutput, "%le\n", sys->s_out[i][j]);
	average += sys->s_out[i][j];
      }
    }

    printf("# Output complete\n");
    printf("# Average intensity: %g\n", average/(1.0*larea));
  }

  fclose(polaroutput);

  return;
}
