/****************************************************************************
 *
 *  polarizer.cpp
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
#include <ostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#define Pi 3.141592653589793

using namespace std;

/*** ADAPT THE FOLLOWING PARAMETERS ***/
int is_nematic=0; // is_nematic sets a constant scalar OP q=0.333
int cut_topbot=0; // ignore cut_topbot sites at entry and exit
double xi_polarizer=0, xi_analyzer=90; // orientation of polarizer and analyzer (deg)
double n_e=2.0, n_o=1.5; // extraordinary and ordinary refraction indices
const int nlambda=1; // number of wavelengths
double lambda[nlambda]={18.0}, weight[nlambda]={1.0}; // wavelengths in l.u. and weights in total sum  

/*** IT SHOULD NOT BE NECESSARY TO MODIFY ANYTHING BELOW THIS LINE ***/
int raydir; // Cartesian direction of incident light, x=0, y=1, z=2
int Lx,Ly,Lz; // box size
int Lxdir,Lydir,Lzdir,Lxsop,Lysop,Lzsop; // box size in input arrays
double ****dir,***sop,*****mueller, ****mueller_sum_tp1, ****mueller_sum_tp2; // director, Mueller matrices
double p1[4][4],***s1,p2[4][4],***s2,***s2_sum, average; // polariser and analyser, Stokes vectors
double ****alp,****bet,***del; // orientation angles of local director, phase shift
// auxiliary variables
char dirfile[250],sopfile[250],outfile[250];
char line[250],line2[250],dummy[250];
string myline;
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
    cout << "# Command line argumets after the executable are: director_filename [scalarOP_filename] ray_direction[x, y OR z]" << endl;
    exit(0);
  }

  // take system dimensions from vtk-header
  sprintf(dirfile,"%s",argv[1]);
  ifstream dirinput(dirfile, ios::in);
  if (!dirinput){
     cout << "Cannot open director input file" << endl;
     exit(0);
  }

  for (int skip=0; skip<4; skip++) {
    getline(dirinput, myline);
  }
  dirinput >> dummy >> Lx >> Ly >> Lz;
  for (int skip=5; skip<10; skip++) {
    getline(dirinput, myline);
  }

  dirinput.close();

  allocate();

  read_data(argc, argv);

  // main loop over light components
  for(a=0; a<=nlambda-1; a++){

    cout << "# Wavelength no " << a+1 << ": lambda=" << lambda[a] << " weight=" << weight[a] << endl;

    initialise_matrices();
    simulate_polarizer();

  }

  output();

  cout << "# Done" << endl;

}

/*****************************************************************************
 *
 *  allocate
 *
 *****************************************************************************/
void allocate(){

  dir = new double***[Lx];
  sop = new double**[Lx];
  mueller = new double****[Lx];

  alp = new double***[Lx];
  bet = new double***[Lx];
  del = new double**[Lx];

  for (i=0; i<Lx; i++){

     dir[i]=new double**[Ly];
     sop[i]=new double*[Ly];
     mueller[i]=new double***[Ly];
     alp[i]=new double**[Ly];
     bet[i]=new double**[Ly];
     del[i]=new double*[Ly];

     for (j=0; j<Ly; j++){

        dir[i][j]=new double*[Lz];
        sop[i][j]=new double[Lz];
        mueller[i][j]=new double**[Lz];
        alp[i][j]=new double*[Lz];
        bet[i][j]=new double*[Lz];
        del[i][j]=new double[Lz];

        for (k=0; k<Lz; k++){

           dir[i][j][k]=new double[3];
           mueller[i][j][k]=new double*[4];
           alp[i][j][k]=new double[3];
           bet[i][j][k]=new double[3];

           for (l=0; l<4; l++){
              mueller[i][j][k][l]=new double[4];
           }

        }
     }
  }

  if(raydir==0){

     s1=new double**[Ly];
     s2=new double**[Ly];
     s2_sum=new double**[Ly];
     mueller_sum_tp1 = new double***[Ly];
     mueller_sum_tp2 = new double***[Ly];

     for(j=0; j<Ly; j++){

        s1[j]=new double*[Lz];
        s2[j]=new double*[Lz];
        s2_sum[j]=new double*[Lz];
        mueller_sum_tp1[j] = new double**[Lz];
        mueller_sum_tp2[j] = new double**[Lz];

        for(k=0; k<Lz; k++){

           s1[j][k]=new double[4];
           s2[j][k]=new double[4];
           s2_sum[j][k]=new double[4];
           mueller_sum_tp1[j][k] = new double*[4];
           mueller_sum_tp2[j][k] = new double*[4];

           for(l=0; l<4; l++){
              mueller_sum_tp1[j][k][l] = new double[4];
              mueller_sum_tp2[j][k][l] = new double[4];
           }
        }
     }
  }

  if(raydir==1){
     s1=new double**[Lx];
     s2=new double**[Lx];
     s2_sum=new double**[Lx];
     mueller_sum_tp1 = new double***[Lx];
     mueller_sum_tp2 = new double***[Lx];


     for(i=0; i<Lx; i++){

        s1[i]=new double*[Lz];
        s2[i]=new double*[Lz];
        s2_sum[i]=new double*[Lz];
        mueller_sum_tp1[i] = new double**[Lz];
        mueller_sum_tp2[i] = new double**[Lz];


        for(k=0; k<Lz; k++){
           s1[i][k]=new double[4];
           s2[i][k]=new double[4];
           s2_sum[i][k]=new double[4];
           mueller_sum_tp1[i][k] = new double*[4];
           mueller_sum_tp2[i][k] = new double*[4];

           for(l=0; l<4; l++){
              mueller_sum_tp1[i][k][l] = new double[4];
              mueller_sum_tp2[i][k][l] = new double[4];
           }
        }
     }
  }

  if(raydir==2){
     s1=new double**[Lx];
     s2=new double**[Lx];
     s2_sum=new double**[Lx];
     mueller_sum_tp1 = new double***[Lx];
     mueller_sum_tp2 = new double***[Lx];


     for(i=0; i<Lx; i++){

        s1[i]=new double*[Ly];
        s2[i]=new double*[Ly];
        s2_sum[i]=new double*[Ly];
        mueller_sum_tp1[i] = new double**[Ly];
        mueller_sum_tp2[i] = new double**[Ly];


        for(j=0; j<Ly; j++){
           s1[i][j]=new double[4];
           s2[i][j]=new double[4];
           s2_sum[i][j]=new double[4];
           mueller_sum_tp1[i][j] = new double*[4];
           mueller_sum_tp2[i][j] = new double*[4];
           for(l=0; l<4; l++){
              mueller_sum_tp1[i][j][l] = new double[4];
              mueller_sum_tp2[i][j][l] = new double[4];
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

  cout << "# Director input\r" << flush; 

  sprintf(dirfile,"%s",argv[1]);
  ifstream dirinput(dirfile, ios::in);
  if (!dirinput){
     cout << "Cannot open director input file" << endl;
     exit(0);
  }

  // skip header lines
  for (int skip=0; skip<9; skip++) {
    getline(dirinput, myline);
  }

  i=-1;
  j=0;
  k=0;

  while(!dirinput.eof()){

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
    dirinput >> dirx >> diry >> dirz;
    dir[i][j][k][0]=dirx;
    dir[i][j][k][1]=diry;
    dir[i][j][k][2]=dirz;
  }

  cout << "# Director input complete" << endl; 

  cout << "# Scalar order parameter input\r" << flush; 

  if(!is_nematic){
    sprintf(sopfile,"%s",argv[2]);
    ifstream sopinput(sopfile, ios::in);
    if (!sopinput){
       cout << "Cannot open scalar order parameter input file" << endl;
       exit(0);
    }
    // skip header lines
    for (int skip=0; skip<4; skip++) {
      getline(sopinput, myline);
    }
    // take system dimensions from vtk-header
    sopinput >> dummy >> Lxsop >> Lysop >> Lzsop;
    // skip header lines
    for (int skip=5; skip<11; skip++) {
      getline(sopinput, myline);
    }

    // compare dimensions for consistency
    if (Lx!=Lxsop || Ly!=Lysop || Lz!=Lzsop) {
      cout << "Inconsistent dimensions in director and scalar OP input" << endl;
      exit(0);
    }

    i=-1;
    j=0;
    k=0;

    while(!sopinput.eof()){

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
      sopinput >> sop0;
      sop[i][j][k]=sop0;
    }
    cout << "# Scalar order parameter input complete" << endl; 
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
    cout << "# Assuming constant scalar order parameter" << endl; 
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

  cout << "# Initialisation\r" << flush; 

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

  cout << "# Initialisation complete" << endl;

}

/*****************************************************************************
 *
 *  simulate_polarizer
 *
 *  Calculates the intensity for a given light component.
 *
 *****************************************************************************/
void simulate_polarizer(){

  cout << "# Simulating polarizer" << endl;

  if(raydir==0){

    for(j=0; j<Ly; j++){
      for(k=0; k<Lz; k++){

        cout << "# j= " << j << " k= " << k << "\r";

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

        cout << "# i= " << i << " k= " << k << "\r";

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

        cout << "# i= " << i << " j= " << j << "\r";

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

  ofstream polaroutput(outfile, ios::out);

  cout << "# Output" << "\r";

  if(raydir==0){

    polaroutput << "# vtk DataFile Version 2.0" << endl;
    polaroutput << "Generated by ludwig extract.c" << endl;
    polaroutput << "ASCII" << endl;
    polaroutput << "DATASET STRUCTURED_POINTS" << endl;
    polaroutput << "DIMENSIONS 1 " << Ly << " " << Lz << endl;
    polaroutput << "ORIGIN 0 0 0" << endl;
    polaroutput << "SPACING 1 1 1" << endl;
    polaroutput << "POINT_DATA " << Ly*Lz << endl;
    polaroutput << "SCALARS Polarizer float 1" << endl;
    polaroutput << "LOOKUP_TABLE default" << endl;

    for(k=0; k<Lz; k++){
      for(j=0; j<Ly; j++){
        polaroutput << s2_sum[j][k][0] << endl;  
      }
    }

    cout << "# Output complete" << endl;
    cout << "# Average intensity: " << average/Ly/Lz/nlambda << endl;

  }

  if(raydir==1){

    polaroutput << "# vtk DataFile Version 2.0" << endl;
    polaroutput << "Generated by ludwig extract.c" << endl;
    polaroutput << "ASCII" << endl;
    polaroutput << "DATASET STRUCTURED_POINTS" << endl;
    polaroutput << "DIMENSIONS " << Lx << " 1 " << Lz << endl;
    polaroutput << "ORIGIN 0 0 0" << endl;
    polaroutput << "SPACING 1 1 1" << endl;
    polaroutput << "POINT_DATA " << Lx*Lz << endl;
    polaroutput << "SCALARS Polarizer float 1" << endl;
    polaroutput << "LOOKUP_TABLE default" << endl;


    for(k=0; k<Lz; k++){
      for(i=0; i<Lx; i++){
        polaroutput << s2_sum[i][k][0] << endl;  
      }
    }

    cout << "# Output complete" << endl;
    cout << "# Average intensity: " << average/Lx/Lz/nlambda << endl;

  }


  if(raydir==2){

    polaroutput << "# vtk DataFile Version 2.0" << endl;
    polaroutput << "Generated by ludwig extract.c" << endl;
    polaroutput << "ASCII" << endl;
    polaroutput << "DATASET STRUCTURED_POINTS" << endl;
    polaroutput << "DIMENSIONS " << Lx << " " << Ly << " 1" << endl;
    polaroutput << "ORIGIN 0 0 0" << endl;
    polaroutput << "SPACING 1 1 1" << endl;
    polaroutput << "POINT_DATA " << Lx*Ly << endl;
    polaroutput << "SCALARS Polarizer float 1" << endl;
    polaroutput << "LOOKUP_TABLE default" << endl;

    for(j=0; j<Ly; j++){
      for(i=0; i<Lx; i++){
        polaroutput << s2_sum[i][j][0] << endl;  
      }
    }

    cout << "# Output complete" << endl;
    cout << "# Average intensity: " << average/Lx/Ly/nlambda << endl;

  }
}
