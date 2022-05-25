/****************************************************************************
 *
 *  polarizer.c
 *
 *  This program simulates the appearance of a liquid-crystalline sample
 *  under polarised light. Director and scalar order parameter data in 
 *  vtk-format is required. The output consists of a file 'polar-...' 
 *  that contains the light intensity in vtk-format.
 * 
 *  To process run
 *
 *  ./polarizer director-file scalarOP-file Cartesian_direction 
 * 
 *  where director-file and scalarOP-file are the data filenames and
 *  Cartesian_direction is x, y, or z and defines the direction of the
 *  incident light.
 *
 *  COMMAND LINE INPUT
 *  (1) the director filename
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
 *  University of Strathclyde, Glasgow, UK
 *
 *  Contributing authors:
 *  Oliver Henrich  (oliver.henrich@strath.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define Pi 3.141592653589793

/*** ADAPT THE FOLLOWING PARAMETERS ***/
int is_nematic=0; // is_nematic sets a constant scalar OP q=0.333
int cut_topbot=0; // ignore cut_topbot sites at entry and exit
double xi_polarizer=0, xi_analyzer=90; // orientation of polarizer and analyzer (deg)
double n_e=2.0, n_o=1.5; // extraordinary and ordinary refraction indices
#define nlambda 3 // number of wavelengths
double lambda[nlambda]={18.0, 20.0, 22.0}, weight[nlambda]={0.2,0.4,0.2}; // wavelengths in l.u. and weights in total sum  

/*** IT SHOULD NOT BE NECESSARY TO MODIFY ANYTHING BELOW THIS LINE ***/
int raydir; // Cartesian direction of incident light, x=0, y=1, z=2
int Lx,Ly,Lz; // box size
int Lxdir,Lydir,Lzdir,Lxsop,Lysop,Lzsop; // box size in input arrays
double ****dir,***sop,*****mueller, ****mueller_sum_tp1, ****mueller_sum_tp2; // director, Mueller matrices
double p1[4][4],***s1,p2[4][4],***s2,***s2_sum, average; // polariser and analyser, Stokes vectors
double ****alp,****bet,***del; // orientation angles of local director, phase shift
// auxiliary variables
#define MAX_LENGTH 256
char dirfile[MAX_LENGTH],sopfile[MAX_LENGTH],outfile[MAX_LENGTH];
char line[MAX_LENGTH],line2[MAX_LENGTH],dummy[MAX_LENGTH];
int a,i,j,k,l,m,n,p;
double dirx,diry,dirz,sop0,Sb,Cb,Sd,Cd;
double fac=0.5, Cx, Sx, Cx2, Sx2;
// functions, see below for further information
void allocate();
void read_data(int, char**);
void initialise_matrices();
void simulate_polarizer();
void output();

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/
int main(int argc, char* argv[]){

  if(is_nematic && argc==3){
    if(*argv[2]=='x') raydir=0;
    if(*argv[2]=='y') raydir=1;
    if(*argv[2]=='z') raydir=2;
  }
  else if(!is_nematic && argc==4){
    if(*argv[3]=='x') raydir=0;
    if(*argv[3]=='y') raydir=1;
    if(*argv[3]=='z') raydir=2;
  }
  else{
    printf("# Command line argumets after the executable are: director_filename [scalarOP_filename] ray_direction[x, y OR z]\n");
    exit(0);
  }

  // take system dimensions from vtk-header
  sprintf(dirfile,"%s",argv[1]);
  FILE *dirinput = fopen(dirfile, "r");
  if (!dirinput){
     printf("Cannot open director input file\n");
     exit(0);
  }

  for (int skip=0; skip<4; skip++) {
    fgets(line, MAX_LENGTH, dirinput);
  }

  fscanf(dirinput, "%s %d %d %d", &dummy, &Lx, &Ly, &Lz);

  for (int skip=5; skip<10; skip++) {
    fgets(line, MAX_LENGTH, dirinput);
  }

  fclose(dirinput);

  allocate();

  read_data(argc, argv);

  // main loop over light components
  for(a=0; a<=nlambda-1; a++){

    printf("# Wavelength no %d: lambda=%d weight=%g\n", a+1, lambda[a], weight[a]);

    initialise_matrices();
    simulate_polarizer();

  }

  output();

  printf("# Done\n");

}

/*****************************************************************************
 *
 *  allocate
 *
 *****************************************************************************/
void allocate(){

  dir = (double****) calloc(Lx, sizeof(double));
  sop = (double***) calloc(Lx, sizeof(double));
  mueller = (double*****) calloc(Lx, sizeof(double));

  alp = (double****) calloc(Lx, sizeof(double));
  bet = (double****) calloc(Lx, sizeof(double));
  del = (double***) calloc(Lx, sizeof(double));

  for (i=0; i<Lx; i++){

    dir[i] = (double***) calloc(Ly, sizeof(double));
    sop[i] = (double**) calloc(Ly, sizeof(double));
    mueller[i] = (double****) calloc(Ly, sizeof(double));
    alp[i] = (double***) calloc(Ly, sizeof(double));
    bet[i] = (double***) calloc(Ly, sizeof(double));
    del[i] = (double**) calloc(Ly, sizeof(double));

    for (j=0; j<Ly; j++){

      dir[i][j] = (double**) calloc(Lz, sizeof(double));
      sop[i][j] = (double*) calloc(Lz, sizeof(double));
      mueller[i][j] = (double***) calloc(Lz, sizeof(double));
      alp[i][j] = (double**) calloc(Lz, sizeof(double));
      bet[i][j] = (double**) calloc(Lz, sizeof(double));
      del[i][j] = (double*) calloc(Lz, sizeof(double));

      for (k=0; k<Lz; k++){

        dir[i][j][k] = (double*) calloc(3, sizeof(double));
        mueller[i][j][k] = (double**) calloc(4, sizeof(double));
        alp[i][j][k] = (double*) calloc(3, sizeof(double));
        bet[i][j][k] = (double*) calloc(3, sizeof(double));

        for (l=0; l<4; l++){
          mueller[i][j][k][l] = (double*) calloc(4, sizeof(double));
        }

      }
    }
  }

  if(raydir==0){

    s1 = (double***) calloc(Ly, sizeof(double));
    s2 = (double***) calloc(Ly, sizeof(double));
    s2_sum = (double***) calloc(Ly, sizeof(double));
    mueller_sum_tp1 = (double****) calloc(Ly, sizeof(double));
    mueller_sum_tp2 = (double****) calloc(Ly, sizeof(double));

    for(j=0; j<Ly; j++){

      s1[j] = (double**) calloc(Lz, sizeof(double));
      s2[j] = (double**) calloc(Lz, sizeof(double));
      s2_sum[j] = (double**) calloc(Lz, sizeof(double));
      mueller_sum_tp1[j] = (double***) calloc(Lz, sizeof(double)); 
      mueller_sum_tp2[j] = (double***) calloc(Lz, sizeof(double));

      for(k=0; k<Lz; k++){

        s1[j][k] = (double*) calloc(4, sizeof(double));
        s2[j][k] = (double*) calloc(4, sizeof(double));
        s2_sum[j][k] = (double*) calloc(4, sizeof(double));
        mueller_sum_tp1[j][k] = (double**) calloc(4, sizeof(double));
        mueller_sum_tp2[j][k] = (double**) calloc(4, sizeof(double));

          for(l=0; l<4; l++){
            mueller_sum_tp1[j][k][l] = (double*) calloc(4, sizeof(double));
            mueller_sum_tp2[j][k][l] = (double*) calloc(4, sizeof(double));
        }
      }
    }
  }

  if(raydir==1){

    s1 = (double***) calloc(Lx, sizeof(double));
    s2 = (double***) calloc(Lx, sizeof(double));
    s2_sum = (double***) calloc(Lx, sizeof(double));
    mueller_sum_tp1 = (double****) calloc(Lx, sizeof(double));
    mueller_sum_tp2 = (double****) calloc(Lx, sizeof(double));

    for(i=0; i<Lx; i++){

      s1[i] = (double**) calloc(Lz, sizeof(double));
      s2[i] = (double**) calloc(Lz, sizeof(double));
      s2_sum[i] = (double**) calloc(Lz, sizeof(double));
      mueller_sum_tp1[i] = (double***) calloc(Lz, sizeof(double));
      mueller_sum_tp2[i] = (double***) calloc(Lz, sizeof(double));

      for(k=0; k<Lz; k++){

        s1[i][k] = (double*) calloc(4, sizeof(double));
        s2[i][k] = (double*) calloc(4, sizeof(double));
        s2_sum[i][k] = (double*) calloc(4, sizeof(double));
        mueller_sum_tp1[i][k] = (double**) calloc(4, sizeof(double));
        mueller_sum_tp2[i][k] = (double**) calloc(4, sizeof(double));

        for(l=0; l<4; l++){

          mueller_sum_tp1[i][k][l] = (double*) calloc(4, sizeof(double));
          mueller_sum_tp2[i][k][l] = (double*) calloc(4, sizeof(double));

        }
      }
    }
  }

  if(raydir==2){

    s1 = (double***) calloc(Lx, sizeof(double));
    s2 = (double***) calloc(Lx, sizeof(double));
    s2_sum = (double***) calloc(Lx, sizeof(double));
    mueller_sum_tp1 = (double****) calloc(Lx, sizeof(double));
    mueller_sum_tp2 = (double****) calloc(Lx, sizeof(double));

    for(i=0; i<Lx; i++){

      s1[i] = (double**) calloc(Ly, sizeof(double));
      s2[i] = (double**) calloc(Ly, sizeof(double));
      s2_sum[i] = (double**) calloc(Ly, sizeof(double));
      mueller_sum_tp1[i] = (double***) calloc(Ly, sizeof(double));
      mueller_sum_tp2[i] = (double***) calloc(Ly, sizeof(double));

      for(j=0; j<Ly; j++){

        s1[i][j] = (double*) calloc(4, sizeof(double));
        s2[i][j] = (double*) calloc(4, sizeof(double));
        s2_sum[i][j] = (double*) calloc(4, sizeof(double));
        mueller_sum_tp1[i][j] = (double**) calloc(4, sizeof(double));
        mueller_sum_tp2[i][j] = (double**) calloc(4, sizeof(double));

        for(l=0; l<4; l++){

          mueller_sum_tp1[i][j][l] = (double*) calloc(4, sizeof(double));
          mueller_sum_tp2[i][j][l] = (double*) calloc(4, sizeof(double));


        }
      }
    }
  }

}

/*****************************************************************************
 *
 *  read_data
 *
 *****************************************************************************/
void read_data(int argc, char** argv){

  printf("# Director input\r");
  fflush(stdout); 

  sprintf(dirfile,"%s",argv[1]);
  FILE *dirinput = fopen(dirfile, "r");
  if (!dirinput){
     printf("Cannot open director input file\n");
     exit(0);
  }

  // skip header lines
  for (int skip=0; skip<9; skip++) {
    fgets(line, MAX_LENGTH, dirinput);
  }

  i=-1;
  j=0;
  k=0;

  while(fgets(line, MAX_LENGTH, dirinput)){

    i++;
    if(i==Lx){
      j++;
      i=0;
      if(j==Ly){
         k++;
         j=0;
         if(k==Lz){break;}
      }
    }
    sscanf(line, "%g %g %g", &dirx, &diry, &dirz);
    dir[i][j][k][0]=dirx;
    dir[i][j][k][1]=diry;
    dir[i][j][k][2]=dirz;

  }

  printf("# Director input complete\n"); 

  printf("# Scalar order parameter input\r");
  fflush(stdout);

  if(!is_nematic){
    sprintf(sopfile,"%s",argv[2]);
    FILE *sopinput = fopen(sopfile, "r");
    if (!sopinput){
       printf("Cannot open scalar order parameter input file\n");
       exit(0);
    }
    // skip header lines
    for (int skip=0; skip<4; skip++) {
      fgets(line, MAX_LENGTH, sopinput);
    }
    // take system dimensions from vtk-header
    fscanf(sopinput, "%s %d %d %d", &dummy, &Lxsop, &Lysop, &Lzsop);

    // skip header lines
    for (int skip=5; skip<11; skip++) {
      fgets(line, MAX_LENGTH, sopinput);
    }

    // compare dimensions for consistency
    if (Lx!=Lxsop || Ly!=Lysop || Lz!=Lzsop) {
      printf("Inconsistent dimensions in director and scalar OP input\n");
      exit(0);
    }

    i=-1;
    j=0;
    k=0;

    while(fgets(line, MAX_LENGTH, sopinput)){

      i++;
      if(i==Lx){
        j++;
        i=0;	
        if(j==Ly){
          k++;
          j=0;
          if(k==Lz){break;}   
        }
      }
      sop0 = atof(line);
      sop[i][j][k]=sop0;

    }
    printf("# Scalar order parameter input complete\n"); 
  }
  else{

    i=-1;
    j=0;
    k=0;

    while(1){

      i++;
      if(i==Lx){
        j++;
        i=0;	
        if(j==Ly){
          k++;
          j=0;
          if(k==Lz){break;}   
        }
      }
      sop[i][j][k]=0.3333333;

    }
    printf("# Assuming constant scalar order parameter\n"); 
  }

  // create output filename
  char *pch;
  pch = strtok(dirfile,"-");

  do{
     pch = strtok(NULL,"-");
     if(pch==NULL) break;
     sprintf(line2,"-%s",pch);
     strcat(line,line2);
  }while(1);

  if(is_nematic){
    sprintf(outfile,"polar-%s%s",argv[2],line);
  }
  else{
    sprintf(outfile,"polar-%s%s",argv[3],line);
  }

}

/*****************************************************************************
 *
 *  initialise_matrices
 *
 *  Initialises the Mueller matrices, Stokes vector and projectors for 
 *  polariser and analyser.
 *
 *****************************************************************************/
void initialise_matrices(){

  printf("# Initialisation\r");
  fflush(stdout);

  if(raydir==0){
     for(j=0; j<Ly; j++){
        for(k=0; k<Lz; k++){
           for(m=0; m<4; m++){
              s2_sum[j][k][m] = 0;
           }
        }
     }
  }

  if(raydir==1){
     for(i=0; i<Lx; i++){
        for(k=0; k<Lz; k++){
           for(m=0; m<4; m++){
              s2_sum[i][k][m] = 0;
           }
        }
     }
  }

  if(raydir==2){
     for(i=0; i<Lx; i++){
        for(j=0; j<Ly; j++){
           for(m=0; m<4; m++){
              s2_sum[i][j][m] = 0;
           }
        }
     }
  }

  for(i=0; i<Lx; i++){
     for(j=0; j<Ly; j++){
        for(k=0; k<Lz; k++){

           // angle between ray direction and local director 

           alp[i][j][k][0]=acos(dir[i][j][k][0]);
           alp[i][j][k][1]=acos(dir[i][j][k][1]);
           alp[i][j][k][2]=acos(dir[i][j][k][2]);

           // angle between axis and projection onto coordinate plane

           bet[i][j][k][0]=atan(dir[i][j][k][2]/dir[i][j][k][1]);
           bet[i][j][k][1]=atan(dir[i][j][k][0]/dir[i][j][k][2]);
           bet[i][j][k][2]=atan(dir[i][j][k][1]/dir[i][j][k][0]);

           del[i][j][k]=2*Pi*sop[i][j][k]*(n_o*n_e/sqrt(n_o*n_o*sin(alp[i][j][k][raydir])*sin(alp[i][j][k][raydir])+\
                                           n_e*n_e*cos(alp[i][j][k][raydir])*cos(alp[i][j][k][raydir]))-n_o)/lambda[a];


           Sd=sin(del[i][j][k]);
           Cd=cos(del[i][j][k]);

           if(raydir==0){
              Sb=sin(2*bet[i][j][k][0]);
              Cb=cos(2*bet[i][j][k][0]);
           }

           if(raydir==1){
              Sb=sin(2*bet[i][j][k][1]);
              Cb=cos(2*bet[i][j][k][1]);
           }

           if(raydir==2){
              Sb=sin(2*bet[i][j][k][2]);
              Cb=cos(2*bet[i][j][k][2]);
           }

           mueller[i][j][k][0][0]=1;
           mueller[i][j][k][0][1]=0;
           mueller[i][j][k][0][2]=0;
           mueller[i][j][k][0][3]=0;

           mueller[i][j][k][1][0]=0;
           mueller[i][j][k][2][0]=0;
           mueller[i][j][k][3][0]=0;

           mueller[i][j][k][1][1]=Cb*Cb+Sb*Sb*Cd;
           mueller[i][j][k][1][2]=Sb*Cb*(1-Cd);
           mueller[i][j][k][1][3]=-Sb*Sd;

           mueller[i][j][k][2][1]=Sb*Cb*(1-Cd);
           mueller[i][j][k][2][2]=Sb*Sb+Cb*Cb*Cd;
           mueller[i][j][k][2][3]=Cb*Sd;

           mueller[i][j][k][3][1]=Sb*Sd;
           mueller[i][j][k][3][2]=-Cb*Sd;
           mueller[i][j][k][3][3]=Cd;

        }
     }
  }

  for(i=0; i<4; i++){
     for(j=0; j<4; j++){
        p1[i][j]=0;
        p2[i][j]=0;
     }
  }

  // polarizer

  Cx=cos(2*xi_polarizer/180*Pi);
  Sx=sin(2*xi_polarizer/180*Pi); 
  Cx2=Cx*Cx;
  Sx2=Sx*Sx;

  p1[0][0]=fac;
  p1[0][1]=fac*Cx;
  p1[0][2]=fac*Sx;
  p1[0][3]=0;

  p1[1][0]=fac*Cx;
  p1[1][1]=fac*Cx2;
  p1[1][2]=fac*Cx*Sx;
  p1[1][3]=0;

  p1[2][0]=fac*Sx;
  p1[2][1]=fac*Sx*Cx;
  p1[2][2]=fac*Sx2;
  p1[2][3]=0;

  p1[3][0]=0;
  p1[3][1]=0;
  p1[3][2]=0;
  p1[3][3]=0;

  // analyzer

  Cx=cos(2*xi_analyzer/180*Pi);
  Sx=sin(2*xi_analyzer/180*Pi); 
  Cx2=Cx*Cx;
  Sx2=Sx*Sx;

  p2[0][0]=fac;
  p2[0][1]=fac*Cx;
  p2[0][2]=fac*Sx;
  p2[0][3]=0;

  p2[1][0]=fac*Cx;
  p2[1][1]=fac*Cx2;
  p2[1][2]=fac*Cx*Sx;
  p2[1][3]=0;

  p2[2][0]=fac*Sx;
  p2[2][1]=fac*Sx*Cx;
  p2[2][2]=fac*Sx2;
  p2[2][3]=0;

  p2[3][0]=0;
  p2[3][1]=0;
  p2[3][2]=0;
  p2[3][3]=0;

  printf("# Initialisation complete\n");

}

/*****************************************************************************
 *
 *  simulate_polarizer
 *
 *  Calculates the intensity for a given light component.
 *
 *****************************************************************************/
void simulate_polarizer(){

  printf("# Simulating polarizer\n");

  if(raydir==0){

    for(j=0; j<Ly; j++){
      for(k=0; k<Lz; k++){

        printf("# j= %d k= %d\r", j, k);

        for(l=0; l<4; l++){
          s1[j][k][l]=0;
          s2[j][k][l]=0;
        }

        // set incident intensity to 1
        s1[j][k][0]=1;

        for(m=0; m<4; m++){
          for(n=0; n<4; n++){
            s2[j][k][m]+=p1[m][n]*s1[j][k][n];

          }
        }

        for(l=0; l<4; l++){
          s1[j][k][l]=s2[j][k][l];
          s2[j][k][l]=0;
        }

        for(i=0+cut_topbot; i<Lx-1-cut_topbot; i++){ 

          if(i==0){
            for(m=0; m<4; m++){
              for(n=0; n<4; n++){
                mueller_sum_tp1[j][k][m][n]=mueller[i][j][k][m][n];
              }
            }
          }

          for(m=0; m<4; m++){
            for(n=0; n<4; n++){

              mueller_sum_tp2[j][k][m][n]=0;

              for(l=0; l<4; l++){
                mueller_sum_tp2[j][k][m][n]+=mueller[i+1][j][k][m][l]*mueller_sum_tp1[j][k][l][n];
              }

            }
          }

          for(m=0; m<4; m++){
            for(n=0; n<4; n++){
              mueller_sum_tp1[j][k][m][n]=mueller_sum_tp2[j][k][m][n];
            }
          }

        }

        for(m=0; m<4; m++){
          for(n=0; n<4; n++){
            s2[j][k][m]+=mueller_sum_tp2[j][k][m][n]*s1[j][k][n];
          }
        }

        for(l=0; l<4; l++){
          s1[j][k][l]=s2[j][k][l];
          s2[j][k][l]=0;
        }

        for(m=0; m<4; m++){
          for(n=0; n<4; n++){
            s2[j][k][m]+=p2[m][n]*s1[j][k][n];
          }
          s2_sum[j][k][m] += weight[a]*s2[j][k][m];
        }

        average+=s2_sum[j][k][0];

      }
    }
  }

  if(raydir==1){

    for(i=0; i<Lx; i++){
      for(k=0; k<Lz; k++){

        printf("# i= %d k= %d\r", i, k);

        for(l=0; l<4; l++){
          s1[i][k][l]=0;
          s2[i][k][l]=0;
        }

        s1[i][k][0]=1;

        for(m=0; m<4; m++){
          for(n=0; n<4; n++){
            s2[i][k][m]+=p1[m][n]*s1[i][k][n];
          }
        }


        for(l=0; l<4; l++){
          s1[i][k][l]=s2[i][k][l];
          s2[i][k][l]=0;
        }

        for(j=0+cut_topbot; j<Ly-1-cut_topbot; j++){ 

          if(j==0){
            for(m=0; m<4; m++){
              for(n=0; n<4; n++){
                mueller_sum_tp1[i][k][m][n]=mueller[i][j][k][m][n];
              }
            }
          }

          for(m=0; m<4; m++){
            for(n=0; n<4; n++){

              mueller_sum_tp2[i][k][m][n]=0;

              for(l=0; l<4; l++){
                mueller_sum_tp2[i][k][m][n]+=mueller[i][j+1][k][m][l]*mueller_sum_tp1[i][k][l][n];
              }

            }
          }

          for(m=0; m<4; m++){
            for(n=0; n<4; n++){
              mueller_sum_tp1[i][k][m][n]=mueller_sum_tp2[i][k][m][n];
            }
          }

        }

        for(m=0; m<4; m++){
          for(n=0; n<4; n++){
            s2[i][k][m]+=mueller_sum_tp2[i][k][m][n]*s1[i][k][n];
          }
        }

        for(l=0; l<4; l++){
          s1[i][k][l]=s2[i][k][l];
          s2[i][k][l]=0;
        }


        for(m=0; m<4; m++){
          for(n=0; n<4; n++){
            s2[i][k][m]+=p2[m][n]*s1[i][k][n];
          }
          s2_sum[i][k][m] += weight[a]*s2[i][k][m];
        }

        average+=s2_sum[i][k][0];

      }
    }
  }

  if(raydir==2){

    for(i=0; i<Lx; i++){
      for(j=0; j<Ly; j++){

        printf("# i= %d j= %d\r", i, j);

        for(l=0; l<4; l++){
          s1[i][j][l]=0;
          s2[i][j][l]=0;
        }

        s1[i][j][0]=1;

        for(m=0; m<4; m++){
          for(n=0; n<4; n++){
            s2[i][j][m]+=p1[m][n]*s1[i][j][n];
          }
        }


        for(l=0; l<4; l++){
          s1[i][j][l]=s2[i][j][l];
          s2[i][j][l]=0;
        }

        for(k=0+cut_topbot; k<Lz-1-cut_topbot; k++){ 

          if(k==cut_topbot){
            for(m=0; m<4; m++){
              for(n=0; n<4; n++){
                mueller_sum_tp1[i][j][m][n]=mueller[i][j][k][m][n];
              }
            }
          }


          for(m=0; m<4; m++){
            for(n=0; n<4; n++){

              mueller_sum_tp2[i][j][m][n]=0;

              for(l=0; l<4; l++){
                mueller_sum_tp2[i][j][m][n]+=mueller[i][j][k+1][m][l]*mueller_sum_tp1[i][j][l][n];
              }

            }
          }

          for(m=0; m<4; m++){
            for(n=0; n<4; n++){
              mueller_sum_tp1[i][j][m][n]=mueller_sum_tp2[i][j][m][n];
            }
          }

        }

        for(m=0; m<4; m++){
          for(n=0; n<4; n++){
            s2[i][j][m]+=mueller_sum_tp2[i][j][m][n]*s1[i][j][n];
          }
        }

        for(l=0; l<4; l++){
          s1[i][j][l]=s2[i][j][l];
          s2[i][j][l]=0;
        }

        for(m=0; m<4; m++){
          for(n=0; n<4; n++){
            s2[i][j][m]+=p2[m][n]*s1[i][j][n];
          }
          s2_sum[i][j][m] += weight[a]*s2[i][j][m];
        }

        average+=s2_sum[i][j][0];

      }
    }

  }
}

/*****************************************************************************
 *
 *  output
 *
 *****************************************************************************/
void output(){

  FILE *polaroutput = fopen(outfile, "w");

  printf("# Output\r");

  if(raydir==0){

    fprintf(polaroutput,"# vtk DataFile Version 2.0\n");
    fprintf(polaroutput,"Generated by ludwig extract.c\n");
    fprintf(polaroutput,"ASCII\n");
    fprintf(polaroutput,"DATASET STRUCTURED_POINTS\n");
    fprintf(polaroutput,"DIMENSIONS 1 %d %d\n", Ly, Lz);
    fprintf(polaroutput,"ORIGIN 0 0 0\n");
    fprintf(polaroutput,"SPACING 1 1 1\n");
    fprintf(polaroutput,"POINT_DATA %d\n", Ly*Lz);
    fprintf(polaroutput,"SCALARS Polarizer float 1\n");
    fprintf(polaroutput,"LOOKUP_TABLE default\n");

    for(k=0; k<Lz; k++){
      for(j=0; j<Ly; j++){
        fprintf(polaroutput, "%le\n", s2_sum[j][k][0]);  
      }
    }

    printf("# Output complete\n");
    printf("# Average intensity: %g\n", average/Ly/Lz/nlambda);

  }

  if(raydir==1){

    fprintf(polaroutput, "# vtk DataFile Version 2.0\n");
    fprintf(polaroutput, "Generated by ludwig extract.c\n");
    fprintf(polaroutput, "ASCII\n");
    fprintf(polaroutput, "DATASET STRUCTURED_POINTS\n");
    fprintf(polaroutput, "DIMENSIONS %d 1 %d\n", Lx, Lz);
    fprintf(polaroutput, "ORIGIN 0 0 0\n");
    fprintf(polaroutput, "SPACING 1 1 1\n");
    fprintf(polaroutput, "POINT_DATA %d\n", Lx*Lz);
    fprintf(polaroutput, "SCALARS Polarizer float 1\n");
    fprintf(polaroutput, "LOOKUP_TABLE default\n");


    for(k=0; k<Lz; k++){
      for(i=0; i<Lx; i++){
        fprintf(polaroutput, "%le\n", s2_sum[i][k][0]);  
      }
    }

    printf("# Output complete\n");
    printf("# Average intensity: %g\n", average/Lx/Lz/nlambda);

  }


  if(raydir==2){

    fprintf(polaroutput, "# vtk DataFile Version 2.0\n");
    fprintf(polaroutput, "Generated by ludwig extract.c\n");
    fprintf(polaroutput, "ASCII\n");
    fprintf(polaroutput, "DATASET STRUCTURED_POINTS\n");
    fprintf(polaroutput, "DIMENSIONS %d %d 1\n", Lx, Ly);
    fprintf(polaroutput, "ORIGIN 0 0 0\n");
    fprintf(polaroutput, "SPACING 1 1 1\n");
    fprintf(polaroutput, "POINT_DATA %d", Lx*Ly);
    fprintf(polaroutput, "SCALARS Polarizer float 1\n");
    fprintf(polaroutput, "LOOKUP_TABLE default\n");

    for(j=0; j<Ly; j++){
      for(i=0; i<Lx; i++){
        fprintf(polaroutput, "%le\n", s2_sum[i][j][0]);  
      }
    }

    printf("# Output complete\n");
    printf("# Average intensity: %g\n", average/Lx/Ly/nlambda);

  }
}
