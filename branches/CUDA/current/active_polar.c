/* Lattice Boltzmann for cholesteric liquid crystal */
/* Compile line on Ultra sun
cc -fast -xO5 lchol_deGe.c -lm -o lchol_deGe
*/
/* Compile line on DEC
cc -fast -O4 lcdg.c -lm -o lcdg
*/

/* Compile line on LINUX
gcc -O4 lchol_deGe.c -lm -o lchol_deGe
*/


//some cpp stuff...
#include <iostream>
#include <fstream>
#include <iomanip>
#include <time.h>
using namespace std;

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*useful constants*/
#define Pi 3.141592653589793
#define TwoPi 6.283185307179586
#define sqr2 1.4142136
#define FourPi 12.566370614359173
#define Inv12Pi 0.026525823848649223
#define Inv4Pi 0.079577471545947668
#define Inv8Pi 0.039788735772973834

/*program parameters*/
#define Lx 1
#define Ly 100
#define Lz 100/*32*/ /*240*/ /*100*/

#define Nmax 20000
int GRAPHICS=1;      /* Graphics on/off */
#define stepskip 200   /*250*/ /* Output graphics every stepskip steps */
#define itimprov 2   /* number of improvement iterations for corrector step */
int runtime=600; // time in seconds after wich it will stop at next save
int started;

/*physical parameters*/
#define temperature 0.5
#define K 0.01 // 0.04  Elastic constant (All)
#define dK -0.00 // delta K difference in elastic constants
#define KLC 0.02 // elastic constant for Liquid christal Free energy

/*active region*/
double ractivesq=40*40.; // radius squared of active region


//Elastic constants for Free energy
#define beta2 0.0 
#define alpha2 -0.1
#define alpha4 0.1

double bodyforce=0.00000; /* body force to set Pouiseuille flow */
double shearv=0.0;

double zeta=0.00; /*active term coming into the pressure tensor */
double zeta0=0.00; /*help variable zeta 0 */
double lambda=0.00; /*active term coming into the Q evolution equation */
double w1=0.00;
                            

#define Gamma 0.3  /*0.33775 diffusion constant */
#define gamma 3.0 /*3.0*/ /*2.695*/ /*2.701*/       /* gamma>2.7 => nematic  */
double aa=4.0;
double phivr=1.0;      /* coupling of hydrodynamics to order parameter */
double phivr_ISO=1.0;
#define xi 2.1 /*0.6*/       
#define tau1 2.5 /* 2.5 proportional to viscosity */
#define dt 1.0
#define densityinit 2.0 /* initial value for the density everywhere */

double namp=0.0;       /* noise amplitude */

/* electric field */
#define EON 0          /* electric field on/off */
double delVz=2.9784;   /* (voltage at z=Lz) - (voltage at z=0) */
double distancesq;
double Q=0.00;

#define epsa 41.4      /* dielectric anisotropy */
#define epsav 9.8      /* average dielectric constant */

/* flexoelectricity */
#define ef0 0.0    /* not implemented yet */
                   /* flexoelectric coefficient, expect ef0 << kappa */

/*boundary conditions--still need a little work*/
/*#define vwtp 0.0   wall velocities y direction */
/*#define vwbt -0.0*/
double vwtp=0.0;/*0.45*1.25;*/
double vwbt=-0.0;/*0.45*1.25;*/
#define BC  0 /*1*/     /* boundary conditions on/off */
#define FIXEDP 0 /*1*/  /* fixed or free boundary conditions for P*/
#define BACKG  0 /* background friction on/off*/  
/* Inital configuration */
#define RANDOM 0 /* Initial configuration with random directions */
#define TWIST 0 /* initial configuration with a twist */
#define ISOTROPIC 0 /*initial configuration isotropic*/
#define VORTEX 0 /*initial configuration vortex*/
#define ASTER 1 /*initial configuration aster*/
#define SPIRAL 0 /*initial configuration as spriral as JF(12)*/
#define TWOSPIRAL 0 /*initial configuration as sprirals or Asters*/
#define NOISY 0.3 /*noisyfy initial config by rotating with amplitude NOISY*/
//several defined not meaningfull, "last" is chosen



#define bcstren 1.0  /* amplitude of harmonic boundary potential */
double wallamp;

#define angztop 0.0 /*88.0 0.0*/
#define angzbot 0.0 /*92.0*/

#define angxytop 90.0
#define angxybot 90.0 /*0.0*/
double Pxtop,Pytop,Pztop,Pxbot,Pybot,Pzbot;
double kappa,tauc,caz;


#define nparticles 0


/*function declarations*/
void writedata(int n);//jens, cpp!!!
void molecularfieldpolar(void);
void equilibriumdist(void);
void initialize(void);
void initializeP(void);
void collisionop(void);
void update0(double (*)[Ly][Lz][15],double (*)[Ly][Lz][15]);
void update(double (*)[Ly][Lz][15],double (*)[Ly][Lz][15]);
void streamfile(void);
double pickphase(double, double);

/*modified numerical recipes routines*/
double gasdev(int);

/*global variables*/
/*lattice Boltzmann variables*/
double fa[Lx][Ly][Lz][15],gaxx[Lx][Ly][Lz][15],gaxy[Lx][Ly][Lz][15];
double gayy[Lx][Ly][Lz][15],gaxz[Lx][Ly][Lz][15],gayz[Lx][Ly][Lz][15];
double fb[Lx][Ly][Lz][15],gbxx[Lx][Ly][Lz][15],gbxy[Lx][Ly][Lz][15];
double Fh[Lx][Ly][Lz][15];
double gbyy[Lx][Ly][Lz][15],gbxz[Lx][Ly][Lz][15],gbyz[Lx][Ly][Lz][15];
double Fc[Lx][Ly][Lz][15],Gxxc[Lx][Ly][Lz][15],Gxyc[Lx][Ly][Lz][15];
double Gyyc[Lx][Ly][Lz][15],Gxzc[Lx][Ly][Lz][15],Gyzc[Lx][Ly][Lz][15];
double freeenergy,freeenergytwist,txx,txy,txz,tyz,tyx,tyy,tzx,tzy,tzz;
double feq[Lx][Ly][Lz][15],gxxeq[Lx][Ly][Lz][15],gxyeq[Lx][Ly][Lz][15];
double gyyeq[Lx][Ly][Lz][15],gxzeq[Lx][Ly][Lz][15],gyzeq[Lx][Ly][Lz][15];
double density[Lx][Ly][Lz],Qxx[Lx][Ly][Lz],Qxy[Lx][Ly][Lz],Qyy[Lx][Ly][Lz];
double Qxz[Lx][Ly][Lz],Qyz[Lx][Ly][Lz],Qzz[Lx][Ly][Lz],u[Lx][Ly][Lz][3];
double Qxxnew[Lx][Ly][Lz],Qxynew[Lx][Ly][Lz],Qyynew[Lx][Ly][Lz];
double Qxznew[Lx][Ly][Lz],Qyznew[Lx][Ly][Lz];
double Qxxold[Lx][Ly][Lz],Qxyold[Lx][Ly][Lz],Qyyold[Lx][Ly][Lz];
double Qxzold[Lx][Ly][Lz],Qyzold[Lx][Ly][Lz];
double Px[Lx][Ly][Lz],Py[Lx][Ly][Lz],Pz[Lx][Ly][Lz];  //polarization field
double Pxold[Lx][Ly][Lz],Pyold[Lx][Ly][Lz],Pzold[Lx][Ly][Lz];
double Pxnew[Lx][Ly][Lz],Pynew[Lx][Ly][Lz],Pznew[Lx][Ly][Lz];
double h[Lx][Ly][Lz][3];
double hnew[Lx][Ly][Lz][3],hold[Lx][Ly][Lz][3];
double hmol[Lx][Ly][Lz][3];
double DEHxx[Lx][Ly][Lz],DEHxy[Lx][Ly][Lz],DEHyy[Lx][Ly][Lz];
double DEHxz[Lx][Ly][Lz],DEHyz[Lx][Ly][Lz];
double DEHxxold[Lx][Ly][Lz],DEHxyold[Lx][Ly][Lz],DEHyyold[Lx][Ly][Lz];
double DEHxzold[Lx][Ly][Lz],DEHyzold[Lx][Ly][Lz];
double DEH1xx[Lx][Ly][Lz],DEH1xy[Lx][Ly][Lz],DEH1yy[Lx][Ly][Lz];
double DEH1xz[Lx][Ly][Lz],DEH1yz[Lx][Ly][Lz];
double DEH2xx[Lx][Ly][Lz],DEH2xy[Lx][Ly][Lz],DEH2yy[Lx][Ly][Lz];
double DEH2xz[Lx][Ly][Lz],DEH2yz[Lx][Ly][Lz];
double DEH3xx[Lx][Ly][Lz],DEH3xy[Lx][Ly][Lz],DEH3yy[Lx][Ly][Lz];
double DEH3xz[Lx][Ly][Lz],DEH3yz[Lx][Ly][Lz];
double DG2xx[Lx][Ly][Lz],DG2yy[Lx][Ly][Lz],DG2xy[Lx][Ly][Lz];
double DG2xz[Lx][Ly][Lz],DG2yz[Lx][Ly][Lz],DG2zz[Lx][Ly][Lz],pG[Lx][Ly][Lz];
double DG3xx[Lx][Ly][Lz],DG3yy[Lx][Ly][Lz],DG3xy[Lx][Ly][Lz];
double DG3xz[Lx][Ly][Lz],DG3yz[Lx][Ly][Lz],DG3zz[Lx][Ly][Lz];
double tauxy[Lx][Ly][Lz],tauxz[Lx][Ly][Lz],tauyz[Lx][Ly][Lz];
double Stressxx[Lx][Ly][Lz],Stressxy[Lx][Ly][Lz],Stressxz[Lx][Ly][Lz];
double Stressyy[Lx][Ly][Lz],Stressyz[Lx][Ly][Lz],Stresszz[Lx][Ly][Lz];
double average_shear_stress=0.0;
double Strainxx[Lx][Ly][Lz],Strainxy[Lx][Ly][Lz],Strainxz[Lx][Ly][Lz];
double Strainyy[Lx][Ly][Lz],Strainyz[Lx][Ly][Lz],Strainzz[Lx][Ly][Lz];
double Ex[Lx][Ly][Lz],Ey[Lx][Ly][Lz],Ez[Lx][Ly][Lz];
double Pdx[Lx][Ly][Lz],Pdy[Lx][Ly][Lz],Pdz[Lx][Ly][Lz];
int pouiseuille;
int pouiseuille1;
int fluid_index[Lx][Ly][Lz];
int e[15][3];
double (*f)[Ly][Lz][15],(*gxx)[Ly][Lz][15],(*gxy)[Ly][Lz][15];
double (*gyy)[Ly][Lz][15],(*gxz)[Ly][Lz][15],(*gyz)[Ly][Lz][15];
double (*fpr)[Ly][Lz][15],(*gxxpr)[Ly][Lz][15],(*gxypr)[Ly][Lz][15];
double (*gyypr)[Ly][Lz][15],(*gxzpr)[Ly][Lz][15],(*gyzpr)[Ly][Lz][15];
double oneplusdtover2tau1,qvisc;
double particle_x[nparticles];
double particle_y[nparticles];
double particle_z[nparticles];
int intx,inty,intz;

FILE *output;
FILE *output1;
FILE *fp;
FILE *fp2;
FILE *fp4;
FILE *fp5;
char name1[40];



//----------------------------------
void loadfluid(){
  fstream in("out.dat",ios::in);
  if(in.good()){    
    for(int i=0; i<Lx; i++) {
      for (int j=0; j<Ly; j++) {
        for (int k=0; k<Lz; k++) {
	  int dummi;
	  in >> dummi;
	  if (dummi != i ){ cerr << "input error" << i<< " "<<j<<" "<<k << endl; exit(1);}
	  in >> dummi;
	  if (dummi != j ){ cerr << "input error" << i<< " "<<j<<" "<<k << endl; exit(1);}
	  in >> dummi;
	  if (dummi != k ){ cerr << "input error" << i<< " "<<j<<" "<<k << endl; exit(1);}

	  in >> Px[i][j][k];
	  in >> Py[i][j][k];
	  in >> Pz[i][j][k];
	  in >> u[i][j][k][0];
	  in >> u[i][j][k][1];
	  in >> u[i][j][k][2];
	  in >> Stressyy[i][j][k];
	  in >> Stressyz[i][j][k];
	  in >> Stresszz[i][j][k];
	  in >> average_shear_stress;
	  in >> h[i][j][k][0];
	  in >> h[i][j][k][1];
	  //  in >> h[i][j][k][2]; not written to output!!!
	}
      }
    }
  }//ende if in.good
}
//----------------------------------


int main(int argc, char** argv)
{
  started=time(0); 
  int i,j,k,n,s,graphstp,improv;
  double (*tmp)[Ly][Lz][15],nextdata;

  scanf("%lf %lf %lf %lf",&zeta,&w1,&bodyforce,&shearv);
  zeta0=zeta;// only neccesary for passive outside

  pouiseuille = 0;
  pouiseuille1=0;

  initialize();
   //other initializations
  graphstp=0; //???
  for(i=0;i<nparticles;i++){ //????
    particle_x[i]=0;
    particle_y[i]=Ly/2.0+i;
    particle_z[i]=Lz/2.0;
  }
  fp4=fopen("disc.out","w");

  loadfluid();


  for (n=1; n<=Nmax; n++) { //main iteration loop


    //not used, nparticles=0 tracer particles
    for(i=0;i<nparticles;i++){
      intx=int(particle_x[i]);
      intx=intx%Lx;//PBC???
      inty=int(particle_y[i]);
      inty=inty%Ly;
      intz=int(particle_z[i]);
      intz=intz%Lz;
      if(intx<0) intx+=Lx;
      if(inty<0) inty+=Ly;
      if(intz<0) intz+=Lz;
      particle_x[i]+=dt*u[intx][inty][intz][0];
      particle_y[i]+=dt*u[intx][inty][intz][1];
      particle_z[i]+=dt*u[intx][inty][intz][2];
    }


      /*uncomment the bit below to use tracers*/

      /*      for(i=0;i<nparticles;i++){
	sprintf(name1,"particle.dat.%d",i);
	fp5=fopen(name1,"a");
	fprintf(fp5,"%16.12f %16.12f %16.12f\n", particle_x[i],particle_y[i],particle_z[i]);
	fflush(fp5);
	fclose(fp5);
      }
      */
      
  // end tracer particles


    //outputing
    if ((n% stepskip ==0) ) {
      fprintf(fp4,"%d %16.10f %16.10f %16.10f %16.10f\n", n, Px[0][0][25],Py[0][0][25],Pz[0][0][25],u[0][0][50][1]);
      fflush(fp4);
    }

    // finite difference, predictor corrector scheme
    for (i=0; i<Lx; i++) {
      for (j=0; j<Ly; j++) {
	for (k=0; k<Lz; k++) {
	  if(fluid_index[i][j][k]==1){
	    Pxold[i][j][k]=Px[i][j][k];
	    Pyold[i][j][k]=Py[i][j][k];
	    Pzold[i][j][k]=Pz[i][j][k];
	  }
	}
      }
    } 

    molecularfieldpolar(); //computes change in director field h(....)

    for (i=0; i<Lx; i++) {
      for (j=0; j<Ly; j++) {
	for (k=0; k<Lz; k++) {
	
	  if(fluid_index[i][j][k]==1){
	    Pxnew[i][j][k]=Pxold[i][j][k]+dt*(h[i][j][k][0]);
	    Pynew[i][j][k]=Pyold[i][j][k]+dt*(h[i][j][k][1]);
	    Pznew[i][j][k]=Pzold[i][j][k]+dt*(h[i][j][k][2]);
	  }

          hold[i][j][k][0]=h[i][j][k][0];
          hold[i][j][k][1]=h[i][j][k][1];
          hold[i][j][k][2]=h[i][j][k][2];

      }
    }
  }
 
    for (i=0; i<Lx; i++) {
      for (j=0; j<Ly; j++) {
	for (k=0; k<Lz; k++) {
	  if(fluid_index[i][j][k]==1){
	    Px[i][j][k]=Pxnew[i][j][k];
	    Py[i][j][k]=Pynew[i][j][k];
	    Pz[i][j][k]=Pznew[i][j][k];
	  }

	  //Free BC ??? check
#if BC
	    if(k==0){
	      Px[i][j][k]=Px[i][j][k+1];
	      Py[i][j][k]=Py[i][j][k+1];
	      Pz[i][j][k]=Pz[i][j][k+1];
	    }
	    if(k==Lz-1){
	      Px[i][j][k]=Px[i][j][k-1];
	      Py[i][j][k]=Py[i][j][k-1];
	      Pz[i][j][k]=Pz[i][j][k-1];
	    }
#endif

#if FIXEDP //fixed BC
	    if(k==0){
	      Px[i][j][k]=Pxbot;
	      Py[i][j][k]=Pybot;
	      Pz[i][j][k]=Pzbot;
	    }
	    if(k==Lz-1){
	      Px[i][j][k]=Pxtop;
	      Py[i][j][k]=Pytop;
	      Pz[i][j][k]=Pztop;
	    }

#endif
      }
    }
    }
    
    /*corrector*/

    for (improv=0; improv<itimprov; improv++) {

      molecularfieldpolar();
       
      for (i=0; i<Lx; i++) {
	for (j=0; j<Ly; j++) {
	  for (k=0; k<Lz; k++) {
	  if(fluid_index[i][j][k]==1){
	    Pxnew[i][j][k]=Pxold[i][j][k]+0.5*dt*(h[i][j][k][0]+
						  hold[i][j][k][0]);
	    Pynew[i][j][k]=Pyold[i][j][k]+0.5*dt*(h[i][j][k][1]+
						  hold[i][j][k][1]);
	    Pznew[i][j][k]=Pzold[i][j][k]+0.5*dt*(h[i][j][k][2]+
						  hold[i][j][k][2]);
	  }
	  }
	}
      }   
      for (i=0; i<Lx; i++) {
	for (j=0; j<Ly; j++) {
	  for (k=0; k<Lz; k++) {
	  if(fluid_index[i][j][k]==1){
	    Px[i][j][k]=Pxnew[i][j][k];
	    Py[i][j][k]=Pynew[i][j][k];
	    Pz[i][j][k]=Pznew[i][j][k];
	  }

#if BC
	    if(k==0){
	      Px[i][j][k]=Px[i][j][k+1];
	      Py[i][j][k]=Py[i][j][k+1];
	      Pz[i][j][k]=Pz[i][j][k+1];
	    }
	    if(k==Lz-1){
	      Px[i][j][k]=Px[i][j][k-1];
	      Py[i][j][k]=Py[i][j][k-1];
	      Pz[i][j][k]=Pz[i][j][k-1];
	    }
#endif

#if FIXEDP
	    if(k==0){
	      Px[i][j][k]=Pxbot;
	      Py[i][j][k]=Pybot;
	      Pz[i][j][k]=Pzbot;
	    }
	    if(k==Lz-1){
	      Px[i][j][k]=Pxtop;
	      Py[i][j][k]=Pytop;
	      Pz[i][j][k]=Pztop;
	    }

#endif
	  }
	}
      }   




    }//end predictor corrector order Parameter dt

    graphstp++;
    
    average_shear_stress=0.0;

    molecularfieldpolar();
    equilibriumdist();

    if ((graphstp%stepskip==0)||(n==1)) {
      streamfile();
      printf("%d\n",n);
      graphstp=0;
      writedata(n);
    }

    if(n==1) {
      vwtp=shearv;
      vwbt=-shearv;
    }

    if (n==1 ) {
      pouiseuille1 = 1;
    }
    collisionop();

    /*predictor*/
    update0(fpr,f);

    tmp=f;
    f=fpr;
    fpr=tmp;

    /*corrector*/
    for (improv=0; improv<itimprov; improv++) {
      molecularfieldpolar();
      equilibriumdist();

      update(f,fpr);

    }
    
#if BACKG    
    //background friction
    for (i=0; i<Lx; i++) {
      for (j=0; j<Ly; j++) {
	for (k=0; k<Lz; k++) {
	  if (((j-Ly/2)*(j-Ly/2)+(k-Lz/2)*(k-Lz/2))> 1.1*ractivesq){
	    u[i][j][k][0]*=0.8;
	    u[i][j][k][1]*=0.8;
	    u[i][j][k][2]*=0.8;
	  }
	}}}
#endif

  }//ende main iteration loop
  fclose(fp4);
}// ende Main


/*----------------------- functions ------------------------------------------*/

void update0(double (*fnew)[Ly][Lz][15],
	    double (*fold)[Ly][Lz][15])
{
  int i,j,k,l,imod,jmod,kmod;
  double rb,Qxxb,Qyyb,Qxyb,Qxzb,Qyzb,tmp,eqrat,vzoutone;

  for (i=0; i<Lx; i++) {
    for (j=0; j<Ly; j++) {
      for (k=0; k<Lz; k++) {

	for (l=0; l<15; l++) {
	  imod=(i-e[l][0]+Lx)%Lx;
	  jmod=(j-e[l][1]+Ly)%Ly;
	  kmod=(k-e[l][2]+Lz)%Lz;
	  fnew[i][j][k][l]=
	    fold[imod][jmod][kmod][l]+dt*Fc[imod][jmod][kmod][l];
	}

	/*B.C.'s*/
	if(pouiseuille1==1){
#if BC
	if(k==0){
	  /* f's */
            fnew[i][j][0][5]= fnew[i][j][0][6];
	  rb=fnew[i][j][0][0]+fnew[i][j][0][1]+fnew[i][j][0][2]+
	    fnew[i][j][0][3]+fnew[i][j][0][4]+
	    2.0*(fnew[i][j][0][6]+fnew[i][j][0][11]+fnew[i][j][0][12]+
	    fnew[i][j][0][13]+fnew[i][j][0][14]);
	  fnew[i][j][0][7]=
	    0.25*(-fnew[i][j][0][1]-fnew[i][j][0][2]+
		 fnew[i][j][0][3]+fnew[i][j][0][4])+
	    0.25*(-fnew[i][j][0][11]+fnew[i][j][0][12]+
		  fnew[i][j][0][14]+rb*vwbt)+0.75*fnew[i][j][0][13];
	  fnew[i][j][0][8]=
	    0.25*(fnew[i][j][0][1]-fnew[i][j][0][2]-
		 fnew[i][j][0][3]+fnew[i][j][0][4])+
	    0.25*(fnew[i][j][0][11]-fnew[i][j][0][12]+
		  fnew[i][j][0][13]+rb*vwbt)+0.75*fnew[i][j][0][14];
	  fnew[i][j][0][9]=
	    0.25*(fnew[i][j][0][1]+fnew[i][j][0][2]-
		 fnew[i][j][0][3]-fnew[i][j][0][4])+
	    0.25*(-fnew[i][j][0][13]+fnew[i][j][0][14]+
		  fnew[i][j][0][12]-rb*vwbt)+0.75*fnew[i][j][0][11];
	  fnew[i][j][0][10]=
	    0.25*(-fnew[i][j][0][1]+fnew[i][j][0][2]+
		 fnew[i][j][0][3]-fnew[i][j][0][4])+
	    0.25*(fnew[i][j][0][11]+fnew[i][j][0][13]-
		  fnew[i][j][0][14]-rb*vwbt)+0.75*fnew[i][j][0][12];

	}
	else if(k==Lz-1){
	  /* f's */
          fnew[i][j][k][6]=fnew[i][j][k][5];
	  rb=fnew[i][j][k][0]+fnew[i][j][k][1]+fnew[i][j][k][2]+
	    fnew[i][j][k][3]+fnew[i][j][k][4]+
	    2.0*(fnew[i][j][k][5]+fnew[i][j][k][7]+fnew[i][j][k][8]+
	    fnew[i][j][k][9]+fnew[i][j][k][10]);
	  fnew[i][j][k][11]=
	    0.25*(-fnew[i][j][k][1]-fnew[i][j][k][2]+
		 fnew[i][j][k][3]+fnew[i][j][k][4])+
	    0.25*(-fnew[i][j][k][7]+fnew[i][j][k][8]+
		  fnew[i][j][k][10]+rb*vwtp)+0.75*fnew[i][j][k][9];
	  fnew[i][j][k][12]=
	    0.25*(fnew[i][j][k][1]-fnew[i][j][k][2]-
		 fnew[i][j][k][3]+fnew[i][j][k][4])+
	    0.25*(fnew[i][j][k][7]-fnew[i][j][k][8]+
		  fnew[i][j][k][9]+rb*vwtp)+0.75*fnew[i][j][k][10];
	  fnew[i][j][k][13]=
	    0.25*(fnew[i][j][k][1]+fnew[i][j][k][2]-
		 fnew[i][j][k][3]-fnew[i][j][k][4])+
	    0.25*(-fnew[i][j][k][9]+fnew[i][j][k][10]+
		  fnew[i][j][k][8]-rb*vwtp)+0.75*fnew[i][j][k][7];
	  fnew[i][j][k][14]=
	    0.25*(-fnew[i][j][k][1]+fnew[i][j][k][2]+
		 fnew[i][j][k][3]-fnew[i][j][k][4])+
	    0.25*(fnew[i][j][k][7]+fnew[i][j][k][9]-
		  fnew[i][j][k][10]-rb*vwtp)+0.75*fnew[i][j][k][8];

	}
#endif
	}
      }
    }
  }
}

void update(double (*fnew)[Ly][Lz][15],
	    double (*fold)[Ly][Lz][15])
{

  int i,j,k,l,imod,jmod,kmod;
  double rb,Qxxb,Qyyb,Qxyb,Qxzb,Qyzb,tmp,vzoutone;

  for (i=0; i<Lx; i++) {
    for (j=0; j<Ly; j++) {
      for (k=0; k<Lz; k++) {

	for (l=0; l<15; l++) {
	  imod=(i-e[l][0]+Lx)%Lx;
	  jmod=(j-e[l][1]+Ly)%Ly;
	  kmod=(k-e[l][2]+Lz)%Lz;
	  fnew[i][j][k][l]=(fold[imod][jmod][kmod][l]+
			0.5*dt*(Fc[imod][jmod][kmod][l]+
				feq[i][j][k][l]/tau1))/oneplusdtover2tau1;
	}

	/*B.C.'s*/
       if(pouiseuille1==1){
#if BC
	if(k==0){
	  /* f's */

          fnew[i][j][0][5]=fnew[i][j][0][6];
	  rb=fnew[i][j][0][0]+fnew[i][j][0][1]+fnew[i][j][0][2]+
	    fnew[i][j][0][3]+fnew[i][j][0][4]+
	    2.0*(fnew[i][j][0][6]+fnew[i][j][0][11]+fnew[i][j][0][12]+
	    fnew[i][j][0][13]+fnew[i][j][0][14]);
	  fnew[i][j][0][7]=
	    0.25*(-fnew[i][j][0][1]-fnew[i][j][0][2]+
		 fnew[i][j][0][3]+fnew[i][j][0][4])+
	    0.25*(-fnew[i][j][0][11]+fnew[i][j][0][12]+
		  fnew[i][j][0][14]+rb*vwbt)+0.75*fnew[i][j][0][13];
	  fnew[i][j][0][8]=
	    0.25*(fnew[i][j][0][1]-fnew[i][j][0][2]-
		 fnew[i][j][0][3]+fnew[i][j][0][4])+
	    0.25*(fnew[i][j][0][11]-fnew[i][j][0][12]+
		  fnew[i][j][0][13]+rb*vwbt)+0.75*fnew[i][j][0][14];
	  fnew[i][j][0][9]=
	    0.25*(fnew[i][j][0][1]+fnew[i][j][0][2]-
		 fnew[i][j][0][3]-fnew[i][j][0][4])+
	    0.25*(-fnew[i][j][0][13]+fnew[i][j][0][14]+
		  fnew[i][j][0][12]-rb*vwbt)+0.75*fnew[i][j][0][11];
	  fnew[i][j][0][10]=
	    0.25*(-fnew[i][j][0][1]+fnew[i][j][0][2]+
		 fnew[i][j][0][3]-fnew[i][j][0][4])+
	    0.25*(fnew[i][j][0][11]+fnew[i][j][0][13]-
		  fnew[i][j][0][14]-rb*vwbt)+0.75*fnew[i][j][0][12];

	}
	else if(k==Lz-1){
	  /* f's */

          fnew[i][j][k][6]=fnew[i][j][k][5];
	  rb=fnew[i][j][k][0]+fnew[i][j][k][1]+fnew[i][j][k][2]+
	    fnew[i][j][k][3]+fnew[i][j][k][4]+
	    2.0*(fnew[i][j][k][5]+fnew[i][j][k][7]+fnew[i][j][k][8]+
	    fnew[i][j][k][9]+fnew[i][j][k][10]);
	  fnew[i][j][k][11]=
	    0.25*(-fnew[i][j][k][1]-fnew[i][j][k][2]+
		 fnew[i][j][k][3]+fnew[i][j][k][4])+
	    0.25*(-fnew[i][j][k][7]+fnew[i][j][k][8]+
		  fnew[i][j][k][10]+rb*vwtp)+0.75*fnew[i][j][k][9];
	  fnew[i][j][k][12]=
	    0.25*(fnew[i][j][k][1]-fnew[i][j][k][2]-
		 fnew[i][j][k][3]+fnew[i][j][k][4])+
	    0.25*(fnew[i][j][k][7]-fnew[i][j][k][8]+
		  fnew[i][j][k][9]+rb*vwtp)+0.75*fnew[i][j][k][10];
	  fnew[i][j][k][13]=
	    0.25*(fnew[i][j][k][1]+fnew[i][j][k][2]-
		 fnew[i][j][k][3]-fnew[i][j][k][4])+
	    0.25*(-fnew[i][j][k][9]+fnew[i][j][k][10]+
		  fnew[i][j][k][8]-rb*vwtp)+0.75*fnew[i][j][k][7];
	  fnew[i][j][k][14]=
	    0.25*(-fnew[i][j][k][1]+fnew[i][j][k][2]+
		 fnew[i][j][k][3]-fnew[i][j][k][4])+
	    0.25*(fnew[i][j][k][7]+fnew[i][j][k][9]-
		  fnew[i][j][k][10]-rb*vwtp)+0.75*fnew[i][j][k][8];


	}
#endif
	}
      }
    }
  }
}


void collisionop(void)
{
  int i,j,k,l;

  for (i=0; i<Lx; i++)
    for (j=0; j<Ly; j++)
      for (k=0; k<Lz; k++)
	for (l=0; l<15; l++) {
	  Fc[i][j][k][l]= (feq[i][j][k][l]-f[i][j][k][l])/tau1;
	  /*          if (pouiseuille==1) {
          Fc[i][j][k][l]+=e[l][1]*0.0*bodyforce*density[i][j][k];
	  }*/
	}
}


void equilibriumdist(void)
{
  double A0,A1,A2,B1,B2,C0,C1,C2,D1,D2;
  double G1xx,G2xx,G2xy,G2xz,G2yz,G1yy,G2yy,G1zz,G2zz;
  double dbdtauxb,dbdtauyb,dbdtauzb;
  double rho,Pxl,Pyl,Pzl,Psquare,Pdoth,usq,udote,omdote;
  double hx,hy,hz;
  double Hxx,Hyy,Hxy,Hxz,Hyz,Qsqxx,Qsqxy,Qsqyy,Qsqzz,Qsqxz,Qsqyz,TrQ2;
  double sigxx,sigyy,sigxy,sigxz,sigyz,sigzz;
  double duxdx,duxdy,duxdz,duydx,duydy,duydz,duzdx,duzdy,duzdz;
  double mDQ4xx,mDQ4xy,mDQ4yy,mDQ4xz,mDQ4yz,mDQ4zz,TrDQI;
  double DQpQDxx,DQpQDxy,DQpQDyy,DQpQDxz,DQpQDyz,DQpQDzz,TrDQpQD;
  double dPxdx,dPydx,dPzdx,dPxdy,dPydy,dPzdy,dPxdz,dPydz,dPzdz;
  double d2Pxdxdx,d2Pxdydy,d2Pxdzdz;
  double d2Pydxdx,d2Pydydy,d2Pydzdz;
  double d2Pzdxdx,d2Pzdydy,d2Pzdzdz;
  int i,j,k,l,iup,idwn,jup,jdwn,kup,kdwn,k1;

  for (i=0; i<Lx; i++) {
    if (i==Lx-1) iup=0; else iup=i+1;
    if (i==0) idwn=Lx-1; else idwn=i-1;
    for (j=0; j<Ly; j++) {
      if (j==Ly-1) jup=0; else jup=j+1;
      if (j==0) jdwn=Ly-1; else jdwn=j-1;
      for (k=0; k<Lz; k++) {
	if (k==Lz-1) kup=0; else kup=k+1;
	if (k==0) kdwn=Lz-1; else kdwn=k-1;
	
	//external passiv region  *active region*
	if (((j-Ly/2)*(j-Ly/2)+(k-Lz/2)*(k-Lz/2))< ractivesq){
	  zeta=zeta0;
	  phivr=1.0;
	}
	else{
	  zeta=0;
	  phivr=phivr_ISO;
	}//end of *active region*
	

	rho=density[i][j][k];
	Pxl=Px[i][j][k];
	Pyl=Py[i][j][k];
	Pzl=Pz[i][j][k];

	hx=hmol[i][j][k][0];
	hy=hmol[i][j][k][1];
	hz=hmol[i][j][k][2];

	dPxdx=(Px[iup][j][k]-Px[idwn][j][k])/2.0;
	dPydx=(Py[iup][j][k]-Py[idwn][j][k])/2.0;
	dPzdx=(Pz[iup][j][k]-Pz[idwn][j][k])/2.0;

	dPxdy=(Px[i][jup][k]-Px[i][jdwn][k])/2.0;
	dPydy=(Py[i][jup][k]-Py[i][jdwn][k])/2.0;
	dPzdy=(Pz[i][jup][k]-Pz[i][jdwn][k])/2.0;

	dPxdz=(Px[i][j][kup]-Px[i][j][kdwn])/2.0;
	dPydz=(Py[i][j][kup]-Py[i][j][kdwn])/2.0;
	dPzdz=(Pz[i][j][kup]-Pz[i][j][kdwn])/2.0;

	d2Pxdxdx=Px[iup][j][k]-2.0*Px[i][j][k]+Px[idwn][j][k];
	d2Pxdydy=Px[i][jup][k]-2.0*Px[i][j][k]+Px[i][jdwn][k];
	d2Pxdzdz=Px[i][j][kup]-2.0*Px[i][j][k]+Px[i][j][kdwn];

	d2Pydxdx=Py[iup][j][k]-2.0*Py[i][j][k]+Py[idwn][j][k];
	d2Pydydy=Py[i][jup][k]-2.0*Py[i][j][k]+Py[i][jdwn][k];
	d2Pydzdz=Py[i][j][kup]-2.0*Py[i][j][k]+Py[i][j][kdwn];

	d2Pzdxdx=Pz[iup][j][k]-2.0*Pz[i][j][k]+Pz[idwn][j][k];
	d2Pzdydy=Pz[i][jup][k]-2.0*Pz[i][j][k]+Pz[i][jdwn][k];
	d2Pzdzdz=Pz[i][j][kup]-2.0*Pz[i][j][k]+Pz[i][j][kdwn];

	/*B.C.; use one-sided derivatives*/
	if(pouiseuille1==1){
#if BC
	if(k==0) {
	  dPxdz= (-3.0*Px[i][j][k]+4.0*Px[i][j][k+1]-Px[i][j][k+2])/2.0;
	  dPydz= (-3.0*Py[i][j][k]+4.0*Py[i][j][k+1]-Py[i][j][k+2])/2.0;
	  dPzdz= (-3.0*Pz[i][j][k]+4.0*Pz[i][j][k+1]-Pz[i][j][k+2])/2.0;

	  d2Pxdzdz= -Px[i][j][k+3]+4.0*Px[i][j][k+2]-
	    5.0*Px[i][j][kup]+2.0*Px[i][j][k];
	  d2Pydzdz= -Py[i][j][k+3]+4.0*Py[i][j][k+2]-
	    5.0*Py[i][j][kup]+2.0*Py[i][j][k];
	  d2Pzdzdz= -Pz[i][j][k+3]+4.0*Pz[i][j][k+2]-
	    5.0*Pz[i][j][kup]+2.0*Pz[i][j][k];

	} 

	else if(k==Lz-1) {
      
	  dPxdz= (3.0*Px[i][j][k]-4.0*Px[i][j][k-1]+Px[i][j][k-2])/2.0;
	  dPydz= (3.0*Py[i][j][k]-4.0*Py[i][j][k-1]+Py[i][j][k-2])/2.0;
	  dPzdz= (3.0*Pz[i][j][k]-4.0*Pz[i][j][k-1]+Pz[i][j][k-2])/2.0;

	  d2Pxdzdz= -Px[i][j][k-3]+4.0*Px[i][j][k-2]-
	    5.0*Px[i][j][kdwn]+2.0*Px[i][j][k];
	  d2Pydzdz= -Py[i][j][k-3]+4.0*Py[i][j][k-2]-
	    5.0*Py[i][j][kdwn]+2.0*Py[i][j][k];
	  d2Pzdzdz= -Pz[i][j][k-3]+4.0*Pz[i][j][k-2]-
	    5.0*Pz[i][j][kdwn]+2.0*Pz[i][j][k];

	}



#endif
	}

	Pdoth=Pxl*hx+Pyl*hy+Pzl*hz;

	sigxx=xi*(Pxl*hx-1.0/3.0*Pdoth);
	sigxy=xi*(Pxl*hy+hx*Pyl)/2.0;
	sigxz=xi*(Pxl*hz+hx*Pzl)/2.0;
	sigyy=xi*(Pyl*hy-1.0/3.0*Pdoth);
	sigyz=xi*(Pyl*hz+hy*Pzl)/2.0;
	sigzz=xi*(Pzl*hz-1.0/3.0*Pdoth);

	Psquare=Pxl*Pxl+Pyl*Pyl+Pzl*Pzl;

	sigxx+=zeta*(Pxl*Pxl-1.0/3.0*Psquare);
	sigxy+=zeta*Pxl*Pyl;
	sigyy+=zeta*(Pyl*Pyl-1.0/3.0*Psquare);
	sigxz+=zeta*Pxl*Pzl;
	sigyz+=zeta*Pyl*Pzl;
	sigzz+=zeta*(Pzl*Pzl-1.0/3.0*Psquare);

	Stressxx[i][j][k]+=-sigxx;
	Stressxy[i][j][k]+=-sigxy;
	Stressxz[i][j][k]+=-sigxz;
	Stressyy[i][j][k]+=-sigyy;
	Stressyz[i][j][k]+=-sigyz;
	Stresszz[i][j][k]+=-sigzz;

	Stressxx[i][j][k]-=K*(dPxdx*dPxdx+dPydx*dPydx+dPzdx*dPzdx);
	Stressxy[i][j][k]-=K*(dPxdx*dPxdy+dPydx*dPydy+dPzdx*dPzdy);
	Stressxz[i][j][k]-=K*(dPxdx*dPxdz+dPydx*dPydz+dPzdx*dPzdz);
	Stressyy[i][j][k]-=K*(dPxdy*dPxdy+dPydy*dPydy+dPzdy*dPzdy);
	Stressyz[i][j][k]-=K*(dPxdy*dPxdz+dPydy*dPydz+dPzdy*dPzdz);
	Stresszz[i][j][k]-=K*(dPxdz*dPxdz+dPydz*dPydz+dPzdz*dPzdz);



	average_shear_stress+=Stressyz[i][j][k];

	duxdx=(u[iup][j][k][0]-u[idwn][j][k][0])/2.0;
	duxdy=(u[i][jup][k][0]-u[i][jdwn][k][0])/2.0;
	duxdz=(u[i][j][kup][0]-u[i][j][kdwn][0])/2.0;
	duydx=(u[iup][j][k][1]-u[idwn][j][k][1])/2.0;
	duydy=(u[i][jup][k][1]-u[i][jdwn][k][1])/2.0;
	duydz=(u[i][j][kup][1]-u[i][j][kdwn][1])/2.0;
	duzdx=(u[iup][j][k][2]-u[idwn][j][k][2])/2.0;
	duzdy=(u[i][jup][k][2]-u[i][jdwn][k][2])/2.0;
	duzdz=(u[i][j][kup][2]-u[i][j][kdwn][2])/2.0;

	dbdtauxb= (tauxy[i][jup][k]-tauxy[i][jdwn][k])/2.0+
	  (tauxz[i][j][kup]-tauxz[i][j][kdwn])/2.0;
	dbdtauyb= -(tauxy[iup][j][k]-tauxy[idwn][j][k])/2.0+
	  (tauyz[i][j][kup]-tauyz[i][j][kdwn])/2.0;
	dbdtauzb= -(tauyz[i][jup][k]-tauyz[i][jdwn][k])/2.0-
	  (tauxz[iup][j][k]-tauxz[idwn][j][k])/2.0;

      /*B.C.; use one-sided derivatives*/
	if(pouiseuille1==1){
#if BC
	if(k==0) {
	  duxdz= (-3.0*u[i][j][k][0]+4.0*u[i][j][k+1][0]-u[i][j][k+2][0])/2.0;
	  duydz= (-3.0*u[i][j][k][1]+4.0*u[i][j][k+1][1]-u[i][j][k+2][1])/2.0;
	  duzdz= (-3.0*u[i][j][k][2]+4.0*u[i][j][k+1][2]-u[i][j][k+2][2])/2.0;

	  dbdtauxb= (tauxy[i][jup][k]-tauxy[i][jdwn][k])/2.0+
	    (-3.0*tauxz[i][j][k]+4.0*tauxz[i][j][k+1]-tauxz[i][j][k+2])/2.0;
	  dbdtauyb= -(tauxy[iup][j][k]-tauxy[idwn][j][k])/2.0+
	    (-3.0*tauyz[i][j][k]+4.0*tauyz[i][j][k+1]-tauyz[i][j][k+2])/2.0;
	}
	else if(k==Lz-1) {
	  duxdz= (3.0*u[i][j][k][0]-4.0*u[i][j][k-1][0]+u[i][j][k-2][0])/2.0;
	  duydz= (3.0*u[i][j][k][1]-4.0*u[i][j][k-1][1]+u[i][j][k-2][1])/2.0;
	  duzdz= (3.0*u[i][j][k][2]-4.0*u[i][j][k-1][2]+u[i][j][k-2][2])/2.0;

	  dbdtauxb= (tauxy[i][jup][k]-tauxy[i][jdwn][k])/2.0+
	    (3.0*tauxz[i][j][k]-4.0*tauxz[i][j][k-1]+tauxz[i][j][k-2])/2.0;
	  dbdtauyb= -(tauxy[iup][j][k]-tauxy[idwn][j][k])/2.0+
	    (3.0*tauyz[i][j][k]-4.0*tauyz[i][j][k-1]+tauyz[i][j][k-2])/2.0;
	}
#endif
	}
	A2= (rho*temperature)/10.0;
	A1= A2;
	A0= rho-14.0*A2;
	B2= rho/24.0;
	B1= 8.0*B2;
	C2= -rho/24.0;
	C1= 2.0*C2;
	C0= -2.0*rho/3.0;
	D2= rho/16.0;
	D1= 8.0*D2;
	G2xx= phivr*(sigxx)/16.0;
	G2yy= phivr*(sigyy)/16.0;
	G2zz= phivr*(sigzz)/16.0;
	G2xy= phivr*(sigxy)/16.0;
	G2xz= phivr*(sigxz)/16.0;
	G2yz= phivr*(sigyz)/16.0;
	G1xx= 8.0*G2xx;
	G1yy= 8.0*G2yy;
	G1zz= 8.0*G2zz;


	usq=u[i][j][k][0]*u[i][j][k][0]+u[i][j][k][1]*u[i][j][k][1]+
	  u[i][j][k][2]*u[i][j][k][2];
	feq[i][j][k][0]=A0+C0*usq;

	for (l=1; l<=6; l++) {
	  udote=u[i][j][k][0]*e[l][0]+u[i][j][k][1]*e[l][1]+
	    u[i][j][k][2]*e[l][2];
	  omdote=dbdtauxb*e[l][0]+dbdtauyb*e[l][1]+dbdtauzb*e[l][2];
	  omdote+=phivr*0.0*(Fh[i][j][k][0]*e[l][0]+Fh[i][j][k][1]*e[l][1]+Fh[i][j][k][2]*e[l][2]);
	  omdote+=e[l][1]*bodyforce;
	  feq[i][j][k][l]=A1+B1*udote+C1*usq+D1*udote*udote+
	    G1xx*e[l][0]*e[l][0]+G1yy*e[l][1]*e[l][1]+G1zz*e[l][2]*e[l][2]+
	    tau1*omdote/3.0;
	}
	for (l=7; l<15; l++) {
	  udote=u[i][j][k][0]*e[l][0]+u[i][j][k][1]*e[l][1]+
	    u[i][j][k][2]*e[l][2];
	  omdote=dbdtauxb*e[l][0]+dbdtauyb*e[l][1]+dbdtauzb*e[l][2];
	  omdote+=phivr*0.0*(Fh[i][j][k][0]*e[l][0]+Fh[i][j][k][1]*e[l][1]+Fh[i][j][k][2]*e[l][2]);
	  omdote+=e[l][1]*bodyforce;
	  feq[i][j][k][l]=A2+B2*udote+C2*usq+D2*udote*udote+
	    G2xx*e[l][0]*e[l][0]+2.0*G2xy*e[l][0]*e[l][1]+
	    2.0*G2xz*e[l][0]*e[l][2]+2.0*G2yz*e[l][1]*e[l][2]+
	    G2yy*e[l][1]*e[l][1]+G2zz*e[l][2]*e[l][2]+tau1*omdote/24.0;
	}
      }
    }
  }
}

void molecularfieldpolar(void)
{
  int i,j,k,iup,idwn,jup,jdwn,kup,kdwn,l;
  double dPxdx,dPydx,dPzdx,dPxdy,dPydy,dPzdy,dPxdz,dPydz,dPzdz;
  double d2Pxdxdx,d2Pxdydy,d2Pxdzdz;
  double d2Pydxdx,d2Pydydy,d2Pydzdz;
  double d2Pzdxdx,d2Pzdydy,d2Pzdzdz;
  double d2Qxxdxdx,d2Qxydxdx,d2Qxzdxdx,d2Qyydxdx,d2Qyzdxdx,d2Qzzdxdx;
  double d2Qxxdydy,d2Qxydydy,d2Qxzdydy,d2Qyydydy,d2Qyzdydy,d2Qzzdydy;
  double d2Qxxdzdz,d2Qxydzdz,d2Qxzdzdz,d2Qyydzdz,d2Qyzdzdz,d2Qzzdzdz;
  double Psquare,divP;
  double duxdx,duxdy,duxdz,duydx,duydy,duydz,duzdx,duzdy,duzdz;
  //dK term
  double divPfield[Lx][Ly][Lz];
  double dPfielddx,dPfielddy,dPfielddz;
  for (i=0; i<Lx; i++) {
    for (j=0; j<Ly; j++) {
      for (k=0; k<Lz; k++) {
	density[i][j][k]=0.0;
	u[i][j][k][0]=u[i][j][k][1]=u[i][j][k][2]=0.0;

	//define Q tensor as auxiliary variable for KLC term
	Qxx[i][j][k]=Px[i][j][k]*Px[i][j][k];
	Qxy[i][j][k]=Px[i][j][k]*Py[i][j][k];
	Qxz[i][j][k]=Px[i][j][k]*Pz[i][j][k];
	Qyy[i][j][k]=Py[i][j][k]*Py[i][j][k];
	Qyz[i][j][k]=Py[i][j][k]*Pz[i][j][k];
	Qzz[i][j][k]=Pz[i][j][k]*Pz[i][j][k];

	for (l=0; l<15; l++) {
	  density[i][j][k] += f[i][j][k][l];
	  u[i][j][k][0] += f[i][j][k][l]*e[l][0];
	  u[i][j][k][1] += f[i][j][k][l]*e[l][1];
	  u[i][j][k][2] += f[i][j][k][l]*e[l][2];
	}
	u[i][j][k][0]=u[i][j][k][0]/density[i][j][k];
	u[i][j][k][1]=u[i][j][k][1]/density[i][j][k];
	u[i][j][k][2]=u[i][j][k][2]/density[i][j][k];
      }
    }
  }

  //dK term
  for (i=0; i<Lx; i++) {
    if (i==Lx-1) iup=0; else iup=i+1;
    if (i==0) idwn=Lx-1; else idwn=i-1;
    for (j=0; j<Ly; j++) {
      if (j==Ly-1) jup=0; else jup=j+1;
      if (j==0) jdwn=Ly-1; else jdwn=j-1;
      for (k=0; k<Lz; k++) {
	if (k==Lz-1) kup=0; else kup=k+1;
	if (k==0) kdwn=Lz-1; else kdwn=k-1;
	dPxdx=(Px[iup][j][k]-Px[idwn][j][k])/2.0;
	dPydy=(Py[i][jup][k]-Py[i][jdwn][k])/2.0;
	dPzdz=(Pz[i][j][kup]-Pz[i][j][kdwn])/2.0;
	divP=dPxdx+dPydy+dPzdz;
	divPfield[i][j][k]=divP;
      }}}


   for (i=0; i<Lx; i++) {
    if (i==Lx-1) iup=0; else iup=i+1;
    if (i==0) idwn=Lx-1; else idwn=i-1;
    for (j=0; j<Ly; j++) {
      if (j==Ly-1) jup=0; else jup=j+1;
      if (j==0) jdwn=Ly-1; else jdwn=j-1;
      for (k=0; k<Lz; k++) {
	if (k==Lz-1) kup=0; else kup=k+1;
	if (k==0) kdwn=Lz-1; else kdwn=k-1;
	int nematic=1;
	//external passiv region  *active region*
	if (((j-Ly/2)*(j-Ly/2)+(k-Lz/2)*(k-Lz/2))< ractivesq){
	  nematic=1;
	}
	else{
	  nematic=-1;
	}//end of *active region*
	if (((j-Ly/2)*(j-Ly/2)+(k-Lz/2)*(k-Lz/2))< 100){//nematic core
	  nematic=1;
	}

	dPxdx=(Px[iup][j][k]-Px[idwn][j][k])/2.0;
	dPydx=(Py[iup][j][k]-Py[idwn][j][k])/2.0; 
	dPzdx=(Pz[iup][j][k]-Pz[idwn][j][k])/2.0;

	dPxdy=(Px[i][jup][k]-Px[i][jdwn][k])/2.0;
	dPydy=(Py[i][jup][k]-Py[i][jdwn][k])/2.0;
	dPzdy=(Pz[i][jup][k]-Pz[i][jdwn][k])/2.0;

	dPxdz=(Px[i][j][kup]-Px[i][j][kdwn])/2.0;
	dPydz=(Py[i][j][kup]-Py[i][j][kdwn])/2.0;
	dPzdz=(Pz[i][j][kup]-Pz[i][j][kdwn])/2.0;

	d2Pxdxdx=Px[iup][j][k]-2.0*Px[i][j][k]+Px[idwn][j][k];
	d2Pxdydy=Px[i][jup][k]-2.0*Px[i][j][k]+Px[i][jdwn][k];
	d2Pxdzdz=Px[i][j][kup]-2.0*Px[i][j][k]+Px[i][j][kdwn];

	d2Pydxdx=Py[iup][j][k]-2.0*Py[i][j][k]+Py[idwn][j][k];
	d2Pydydy=Py[i][jup][k]-2.0*Py[i][j][k]+Py[i][jdwn][k];
	d2Pydzdz=Py[i][j][kup]-2.0*Py[i][j][k]+Py[i][j][kdwn];

	d2Pzdxdx=Pz[iup][j][k]-2.0*Pz[i][j][k]+Pz[idwn][j][k];
	d2Pzdydy=Pz[i][jup][k]-2.0*Pz[i][j][k]+Pz[i][jdwn][k];
	d2Pzdzdz=Pz[i][j][kup]-2.0*Pz[i][j][k]+Pz[i][j][kdwn];

	d2Qxxdxdx=Qxx[iup][j][k]-2.0*Qxx[i][j][k]+Qxx[idwn][j][k];
	d2Qxxdydy=Qxx[i][jup][k]-2.0*Qxx[i][j][k]+Qxx[i][jdwn][k];
	d2Qxxdzdz=Qxx[i][j][kup]-2.0*Qxx[i][j][k]+Qxx[i][j][kdwn];

	d2Qxydxdx=Qxy[iup][j][k]-2.0*Qxy[i][j][k]+Qxy[idwn][j][k];
	d2Qxydydy=Qxy[i][jup][k]-2.0*Qxy[i][j][k]+Qxy[i][jdwn][k];
	d2Qxydzdz=Qxy[i][j][kup]-2.0*Qxy[i][j][k]+Qxy[i][j][kdwn];

	d2Qxzdxdx=Qxz[iup][j][k]-2.0*Qxz[i][j][k]+Qxz[idwn][j][k];
	d2Qxzdydy=Qxz[i][jup][k]-2.0*Qxz[i][j][k]+Qxz[i][jdwn][k];
	d2Qxzdzdz=Qxz[i][j][kup]-2.0*Qxz[i][j][k]+Qxz[i][j][kdwn];

	d2Qyydxdx=Qyy[iup][j][k]-2.0*Qyy[i][j][k]+Qyy[idwn][j][k];
	d2Qyydydy=Qyy[i][jup][k]-2.0*Qyy[i][j][k]+Qyy[i][jdwn][k];
	d2Qyydzdz=Qyy[i][j][kup]-2.0*Qyy[i][j][k]+Qyy[i][j][kdwn];

	d2Qyzdxdx=Qyz[iup][j][k]-2.0*Qyz[i][j][k]+Qyz[idwn][j][k];
	d2Qyzdydy=Qyz[i][jup][k]-2.0*Qyz[i][j][k]+Qyz[i][jdwn][k];
	d2Qyzdzdz=Qyz[i][j][kup]-2.0*Qyz[i][j][k]+Qyz[i][j][kdwn];

	d2Qzzdxdx=Qzz[iup][j][k]-2.0*Qzz[i][j][k]+Qzz[idwn][j][k];
	d2Qzzdydy=Qzz[i][jup][k]-2.0*Qzz[i][j][k]+Qzz[i][jdwn][k];
	d2Qzzdzdz=Qzz[i][j][kup]-2.0*Qzz[i][j][k]+Qzz[i][j][kdwn];

	//dK term
	dPfielddx=(divPfield[iup][j][k]-divPfield[idwn][j][k])/2.0;
	dPfielddy=(divPfield[i][jup][k]-divPfield[i][jdwn][k])/2.0;
	dPfielddz=(divPfield[i][j][kup]-divPfield[i][j][kdwn])/2.0;
	//end dK term

	//KLC terms:
	double PdPx=Px[i][j][k]*dPxdx+Py[i][j][k]*dPydx+Pz[i][j][k]*dPzdx;
	double PdPy=Px[i][j][k]*dPxdy+Py[i][j][k]*dPydy+Pz[i][j][k]*dPzdy;
	double PdPz=Px[i][j][k]*dPxdz+Py[i][j][k]*dPydz+Pz[i][j][k]*dPzdz;
	double PddP=Px[i][j][k]*(d2Pxdxdx+d2Pxdydy+d2Pxdzdz)+\
	            Py[i][j][k]*(d2Pydxdx+d2Pydydy+d2Pydzdz)+\
	            Pz[i][j][k]*(d2Pzdxdx+d2Pzdydy+d2Pzdzdz);



	/*B.C.; use one-sided derivatives*/
	if(pouiseuille1==1){
#if BC
	if(k==0) {
	  dPxdz= (-3.0*Px[i][j][k]+4.0*Px[i][j][k+1]-Px[i][j][k+2])/2.0;
	  dPydz= (-3.0*Py[i][j][k]+4.0*Py[i][j][k+1]-Py[i][j][k+2])/2.0;
	  dPzdz= (-3.0*Pz[i][j][k]+4.0*Pz[i][j][k+1]-Pz[i][j][k+2])/2.0;

	  d2Pxdzdz= -Px[i][j][k+3]+4.0*Px[i][j][k+2]-
	    5.0*Px[i][j][kup]+2.0*Px[i][j][k];
	  d2Pydzdz= -Py[i][j][k+3]+4.0*Py[i][j][k+2]-
	    5.0*Py[i][j][kup]+2.0*Py[i][j][k];
	  d2Pzdzdz= -Pz[i][j][k+3]+4.0*Pz[i][j][k+2]-
	    5.0*Pz[i][j][kup]+2.0*Pz[i][j][k];

	  d2Qxxdzdz= -Qxx[i][j][k+3]+4.0*Qxx[i][j][k+2]-
	    5.0*Qxx[i][j][kup]+2.0*Qxx[i][j][k];
	  d2Qxydzdz= -Qxy[i][j][k+3]+4.0*Qxy[i][j][k+2]-
	    5.0*Qxy[i][j][kup]+2.0*Qxy[i][j][k];
	  d2Qxzdzdz= -Qxz[i][j][k+3]+4.0*Qxz[i][j][k+2]-
	    5.0*Qxz[i][j][kup]+2.0*Qxz[i][j][k];
	  d2Qyydzdz= -Qyy[i][j][k+3]+4.0*Qyy[i][j][k+2]-
	    5.0*Qyy[i][j][kup]+2.0*Qyy[i][j][k];
	  d2Qyzdzdz= -Qyz[i][j][k+3]+4.0*Qyz[i][j][k+2]-
	    5.0*Qyz[i][j][kup]+2.0*Qyz[i][j][k];
	  d2Qzzdzdz= -Qzz[i][j][k+3]+4.0*Qzz[i][j][k+2]-
	    5.0*Qzz[i][j][kup]+2.0*Qzz[i][j][k];

#if dK

	  This does not work yet !!

#endif


	}

	else if(k==Lz-1) {
      
	  dPxdz= (3.0*Px[i][j][k]-4.0*Px[i][j][k-1]+Px[i][j][k-2])/2.0;
	  dPydz= (3.0*Py[i][j][k]-4.0*Py[i][j][k-1]+Py[i][j][k-2])/2.0;
	  dPzdz= (3.0*Pz[i][j][k]-4.0*Pz[i][j][k-1]+Pz[i][j][k-2])/2.0;

	  d2Pxdzdz= -Px[i][j][k-3]+4.0*Px[i][j][k-2]-
	    5.0*Px[i][j][kdwn]+2.0*Px[i][j][k];
	  d2Pydzdz= -Py[i][j][k-3]+4.0*Py[i][j][k-2]-
	    5.0*Py[i][j][kdwn]+2.0*Py[i][j][k];
	  d2Pzdzdz= -Pz[i][j][k-3]+4.0*Pz[i][j][k-2]-
	    5.0*Pz[i][j][kdwn]+2.0*Pz[i][j][k];

	  d2Qxxdzdz= -Qxx[i][j][k-3]+4.0*Qxx[i][j][k-2]-
	    5.0*Qxx[i][j][kdwn]+2.0*Qxx[i][j][k];
	  d2Qxxdzdz= -Qxy[i][j][k-3]+4.0*Qxy[i][j][k-2]-
	    5.0*Qxy[i][j][kdwn]+2.0*Qxy[i][j][k];
	  d2Qxxdzdz= -Qxz[i][j][k-3]+4.0*Qxz[i][j][k-2]-
	    5.0*Qxz[i][j][kdwn]+2.0*Qxz[i][j][k];
	  d2Qyydzdz= -Qyy[i][j][k-3]+4.0*Qyy[i][j][k-2]-
	    5.0*Qyy[i][j][kdwn]+2.0*Qyy[i][j][k];
	  d2Qyzdzdz= -Qyz[i][j][k-3]+4.0*Qyz[i][j][k-2]-
	    5.0*Qyz[i][j][kdwn]+2.0*Qyz[i][j][k];
	  d2Qzzdzdz= -Qzz[i][j][k-3]+4.0*Qzz[i][j][k-2]-
	    5.0*Qzz[i][j][kdwn]+2.0*Qzz[i][j][k];

#if dK

	  This does not work yet !!

#endif


	}

#endif
	}

	Psquare=Px[i][j][k]*Px[i][j][k]+Py[i][j][k]*Py[i][j][k]+Pz[i][j][k]*Pz[i][j][k];
	divP=dPxdx+dPydy+dPzdz;

	h[i][j][k][0]=(K+dK)*(d2Pxdxdx+d2Pxdydy+d2Pxdzdz);
	h[i][j][k][1]=(K+dK)*(d2Pydxdx+d2Pydydy+d2Pydzdz);
	h[i][j][k][2]=(K+dK)*(d2Pzdxdx+d2Pzdydy+d2Pzdzdz);
	//dK term
	h[i][j][k][0]-=dK*dPfielddx;
	h[i][j][k][1]-=dK*dPfielddy;
	h[i][j][k][2]-=dK*dPfielddz;
	//end dK term

	//KLC stuff:

	h[i][j][k][0]+=2*KLC*(Px[i][j][k]*(d2Qxxdxdx+d2Qxxdydy+d2Qxxdzdz)+\
			      Py[i][j][k]*(d2Qxydxdx+d2Qxydydy+d2Qxydzdz)+\
			      Pz[i][j][k]*(d2Qxzdxdx+d2Qxzdydy+d2Qxzdzdz));

	h[i][j][k][1]+=2*KLC*(Px[i][j][k]*(d2Qxydxdx+d2Qxydydy+d2Qxydzdz)+\
			      Py[i][j][k]*(d2Qyydxdx+d2Qyydydy+d2Qyydzdz)+\
			      Pz[i][j][k]*(d2Qyzdxdx+d2Qyzdydy+d2Qyzdzdz));

	h[i][j][k][2]+=2*KLC*(Px[i][j][k]*(d2Qxzdxdx+d2Qxzdydy+d2Qxzdzdz)+\
			      Py[i][j][k]*(d2Qyzdxdx+d2Qyzdydy+d2Qyzdzdz)+\
			      Pz[i][j][k]*(d2Qzzdxdx+d2Qzzdydy+d2Qzzdzdz));

	//... end KLC


	h[i][j][k][0]+=2.0*beta2*(-Px[i][j][k]*divP+Px[i][j][k]*dPxdx+Py[i][j][k]*dPydx+Pz[i][j][k]*dPzdx);
	h[i][j][k][1]+=2.0*beta2*(-Py[i][j][k]*divP+Px[i][j][k]*dPxdy+Py[i][j][k]*dPydy+Pz[i][j][k]*dPzdy);
	h[i][j][k][2]+=2.0*beta2*(-Pz[i][j][k]*divP+Px[i][j][k]*dPxdz+Py[i][j][k]*dPydz+Pz[i][j][k]*dPzdz);

	h[i][j][k][0]+=-nematic*alpha2*Px[i][j][k]-alpha4*Psquare*Px[i][j][k];
	h[i][j][k][1]+=-nematic*alpha2*Py[i][j][k]-alpha4*Psquare*Py[i][j][k];
	h[i][j][k][2]+=-nematic*alpha2*Pz[i][j][k]-alpha4*Psquare*Pz[i][j][k];


#if EON


	h[i][j][k][0]+=0;
	h[i][j][k][1]+=(Ey[i][j][k]*Py[i][j][k]+Ez[i][j][k]*Pz[i][j][k])*Ey[i][j][k];
	h[i][j][k][2]+=(Ey[i][j][k]*Py[i][j][k]+Ez[i][j][k]*Pz[i][j][k])*Ez[i][j][k];


#endif

	hmol[i][j][k][0]=h[i][j][k][0];
	hmol[i][j][k][1]=h[i][j][k][1];
	hmol[i][j][k][2]=h[i][j][k][2];

	Fh[i][j][k][0]=h[i][j][k][0]*dPxdx+h[i][j][k][1]*dPydx+h[i][j][k][2]*dPzdx;
	Fh[i][j][k][1]=h[i][j][k][0]*dPxdy+h[i][j][k][1]*dPydy+h[i][j][k][2]*dPzdy;
	Fh[i][j][k][2]=h[i][j][k][0]*dPxdz+h[i][j][k][1]*dPydz+h[i][j][k][2]*dPzdz;

	h[i][j][k][0]-=u[i][j][k][0]*dPxdx+u[i][j][k][1]*dPxdy+u[i][j][k][2]*dPxdz;
	h[i][j][k][1]-=u[i][j][k][0]*dPydx+u[i][j][k][1]*dPydy+u[i][j][k][2]*dPydz;
	h[i][j][k][2]-=u[i][j][k][0]*dPzdx+u[i][j][k][1]*dPzdy+u[i][j][k][2]*dPzdz;

	h[i][j][k][0]-=w1*(Px[i][j][k]*dPxdx+Py[i][j][k]*dPxdy+Pz[i][j][k]*dPxdz);
	h[i][j][k][1]-=w1*(Px[i][j][k]*dPydx+Py[i][j][k]*dPydy+Pz[i][j][k]*dPydz);
	h[i][j][k][2]-=w1*(Px[i][j][k]*dPzdx+Py[i][j][k]*dPzdy+Pz[i][j][k]*dPzdz);

	duxdx=(u[iup][j][k][0]-u[idwn][j][k][0])/2.0;
	duxdy=(u[i][jup][k][0]-u[i][jdwn][k][0])/2.0;
	duxdz=(u[i][j][kup][0]-u[i][j][kdwn][0])/2.0;
	duydx=(u[iup][j][k][1]-u[idwn][j][k][1])/2.0;
	duydy=(u[i][jup][k][1]-u[i][jdwn][k][1])/2.0;
	duydz=(u[i][j][kup][1]-u[i][j][kdwn][1])/2.0;
	duzdx=(u[iup][j][k][2]-u[idwn][j][k][2])/2.0;
	duzdy=(u[i][jup][k][2]-u[i][jdwn][k][2])/2.0;
	duzdz=(u[i][j][kup][2]-u[i][j][kdwn][2])/2.0;

      /*B.C.; use one-sided derivatives*/
#if BC
	if(k==0) {
	  duxdz= (-3.0*u[i][j][k][0]+4.0*u[i][j][k+1][0]-u[i][j][k+2][0])/2.0;
	  duydz= (-3.0*u[i][j][k][1]+4.0*u[i][j][k+1][1]-u[i][j][k+2][1])/2.0;
	  duzdz= (-3.0*u[i][j][k][2]+4.0*u[i][j][k+1][2]-u[i][j][k+2][2])/2.0;

	}
	else if(k==Lz-1) {
	  duxdz= (3.0*u[i][j][k][0]-4.0*u[i][j][k-1][0]+u[i][j][k-2][0])/2.0;
	  duydz= (3.0*u[i][j][k][1]-4.0*u[i][j][k-1][1]+u[i][j][k-2][1])/2.0;
	  duzdz= (3.0*u[i][j][k][2]-4.0*u[i][j][k-1][2]+u[i][j][k-2][2])/2.0;

	}
#endif

	Stressxx[i][j][k]=2.0/3.0*tau1*(2.0*duxdx);
	Stressxy[i][j][k]=2.0/3.0*tau1*(duxdy+duydx);
	Stressxz[i][j][k]=2.0/3.0*tau1*(duxdz+duzdx);
	Stressyy[i][j][k]=2.0/3.0*tau1*(2.0*duydy);
	Stressyz[i][j][k]=2.0/3.0*tau1*(duydz+duzdy);
	Stresszz[i][j][k]=2.0/3.0*tau1*(2.0*duzdz);

	average_shear_stress+=2.0/3.0*tau1*(duydz+duzdy);

	Strainxx[i][j][k]=2.0*duxdx;
	Strainxy[i][j][k]=duxdy+duydx;
	Strainxz[i][j][k]=duxdz+duzdx;
	Strainyy[i][j][k]=2.0*duydy;
	Strainyz[i][j][k]=duydz+duzdy;
	Strainzz[i][j][k]=2.0*duzdz;	

	h[i][j][k][0]-=(duydx-duxdy)/2.0*Py[i][j][k]+(duzdx-duxdz)/2.0*Pz[i][j][k];
	h[i][j][k][1]-=(duxdy-duydx)/2.0*Px[i][j][k]+(duzdy-duydz)/2.0*Pz[i][j][k];
	h[i][j][k][2]-=(duxdz-duzdx)/2.0*Px[i][j][k]+(duydz-duzdy)/2.0*Py[i][j][k];
	
	h[i][j][k][0]+=xi*(duxdx*Px[i][j][k]+(duydx+duxdy)/2.0*Py[i][j][k]+(duzdx+duxdz)/2.0*Pz[i][j][k]);
	h[i][j][k][1]+=xi*((duxdy+duydx)/2.0*Px[i][j][k]+duydy*Py[i][j][k]+(duzdy+duydz)/2.0*Pz[i][j][k]);
	h[i][j][k][2]+=xi*((duxdz+duzdx)/2.0*Px[i][j][k]+(duydz+duzdy)/2.0*Py[i][j][k]+duzdz*Pz[i][j][k]);





	/* work out antisymmetric part of the stress tensor */
	tauxy[i][j][k]= phivr*
	  (Px[i][j][k]*hmol[i][j][k][1]-Py[i][j][k]*hmol[i][j][k][0])/2.0;
	tauxz[i][j][k]= phivr*
	  (Px[i][j][k]*hmol[i][j][k][2]-Pz[i][j][k]*hmol[i][j][k][0])/2.0;
	tauyz[i][j][k]= phivr*
	  (Py[i][j][k]*hmol[i][j][k][2]-Pz[i][j][k]*hmol[i][j][k][1])/2.0;

	Stressxy[i][j][k]+=tauxy[i][j][k];
	Stressxz[i][j][k]+=tauxz[i][j][k];
	Stressyz[i][j][k]+=tauyz[i][j][k];

	average_shear_stress+=tauyz[i][j][k];

      }
    }
   }

}




void initialize(void)
{
  e[0][0]= 0;
  e[0][1]= 0;
  e[0][2]= 0;

  e[1][0]= 1;
  e[1][1]= 0;
  e[1][2]= 0;

  e[2][0]= 0;
  e[2][1]= 1;
  e[2][2]= 0;

  e[3][0]= -1;
  e[3][1]= 0;
  e[3][2]= 0;

  e[4][0]= 0;
  e[4][1]= -1;
  e[4][2]= 0;

  e[5][0]= 0;
  e[5][1]= 0;
  e[5][2]= 1;

  e[6][0]= 0;
  e[6][1]= 0;
  e[6][2]= -1;

  e[7][0]= 1;
  e[7][1]= 1;
  e[7][2]= 1;

  e[8][0]= -1;
  e[8][1]= 1;
  e[8][2]= 1;

  e[9][0]= -1;
  e[9][1]= -1;
  e[9][2]= 1;

  e[10][0]= 1;
  e[10][1]= -1;
  e[10][2]= 1;

  e[11][0]= 1;
  e[11][1]= 1;
  e[11][2]= -1;

  e[12][0]= -1;
  e[12][1]= 1;
  e[12][2]= -1;

  e[13][0]= -1;
  e[13][1]= -1;
  e[13][2]= -1;

  e[14][0]= 1;
  e[14][1]= -1;
  e[14][2]= -1;

 gasdev(1);

  f=fa;

  fpr=fb;

  oneplusdtover2tau1=1.0+0.5*dt/tau1;
 
  Pxtop=sin(angztop/180.0*Pi)*cos(angxytop/180.0*Pi);
  Pytop=sin(angztop/180.0*Pi)*sin(angxytop/180.0*Pi);
  Pztop=cos(angztop/180.0*Pi);

  Pxtop=Pxtop*sqrt(-alpha2/alpha4);
  Pytop=Pytop*sqrt(-alpha2/alpha4);
  Pztop=Pztop*sqrt(-alpha2/alpha4);

  Pxbot=sin(angzbot/180.0*Pi)*cos(angxybot/180.0*Pi);
  Pybot=sin(angzbot/180.0*Pi)*sin(angxybot/180.0*Pi);
  Pzbot=cos(angzbot/180.0*Pi);

  Pxbot=Pxbot*sqrt(-alpha2/alpha4);
  Pybot=Pybot*sqrt(-alpha2/alpha4);
  Pzbot=Pzbot*sqrt(-alpha2/alpha4);

  initializeP();

}


void initializeP(void)
{
  int i,j,k,l,ic,jc,kc;
  double phase,phase2,amplitude,distance;

  for (i=0; i<Lx; i++) {
    for (j=0; j<Ly; j++) {
      for (k=0; k< Lz; k++) {

#if EON
	double yc=(double(j)-double(Ly)/2);//distance from center
	double zc=(double(k)-double(Lz)/2);
	distancesq=yc*yc+zc*zc+0.1;
	Ey[i][j][k]=Q/sqrt(distancesq)*yc;
	Ez[i][j][k]=Q/sqrt(distancesq)*zc;
	//	if (distancesq<0.8*ractivesq) {
	//	Ey[i][j][k]=0;
	//	Ez[i][j][k]=0;
	//	}
#endif

	density[i][j][k]=densityinit;

	amplitude=0.03333333;

#if RANDOM

	phase= 2.0/5.0*Pi*(0.5-drand48());
	phase2= Pi/2.0+Pi/5.0*(0.5-drand48());
		if (k<2 || k>Lz-3) {
	  phase2=Pi/2.0;
	  phase=0.0;
	  }
#endif


#if TWIST
	       phase= Pi/180.0*(angxytop-(angxytop-angxybot)/(Lz-1.0)*k);
	       phase2= Pi/180.0*(angztop-(angztop-(angzbot))/(Lz-1.0)*k);
	         if((k==(Lz/2.0))||(k==((Lz-1)/2.0))) phase2=-Pi/180.*80.0;
	       if(j>Ly/2.0){
		 if((k==(Lz/2.0))||(k==((Lz-1)/2.0))) phase2=Pi/180.*80.0;
		 }

#endif


	       Px[i][j][k]=0.0*sin(phase2)*cos(phase);
	       Py[i][j][k]=sin(phase2)*sin(phase);
	       Pz[i][j][k]=cos(phase2);


#if ISOTROPIC
	Px[i][j][k]=0;
	Py[i][j][k]=0;
	Pz[i][j][k]=0;
#endif
	       fluid_index[i][j][k]=1;
#if VORTEX
	       /* the following part initialises the system with an vortex*/
	       ic=0;
	       jc=Ly/2;
	       kc=Lz/2;
	       distance=sqrt(pow(i-ic,2.0)+pow(j-jc,2.0)+pow(k-kc,2.0));
	       if(distance>0.1){
		 Px[i][j][k]=(i-ic)/distance;
		 Py[i][j][k]=-(k-kc)/distance;
		 Pz[i][j][k]=(j-jc)/distance;

	       }
	       if(distance<2.1){
		 fluid_index[i][j][k]=1;
	       }
	     
#endif
#if SPIRAL
	       /* the following part initialises the system with a spiral, see PRL 2004, eq 12*/
	       double psi0=Pi-0.5*acos(1./xi); //inward spiral, anticlockwise as in Fig1
	       // in principle four solutions: +/-( Pi/0 - -0.5*acos(1./xi))
	       //psi0=Pi-0.5*acos(1./xi); // inward spiral
		 ic=0;
	       jc=Ly/2;
	       kc=Lz/2;
	       distance=sqrt(pow(i-ic,2.0)+pow(j-jc,2.0)+pow(k-kc,2.0));
	       if(distance>0.1){
		 Px[i][j][k]=(i-ic)/distance;
		 Py[i][j][k]=cos(psi0)*(j-jc)/distance-sin(psi0)*(k-kc)/distance;
		 Pz[i][j][k]=sin(psi0)*(j-jc)/distance+cos(psi0)*(k-kc)/distance;

	       }
	       if(distance<2.1){
		 fluid_index[i][j][k]=1;
	       }
	     
#endif


#if TWOSPIRAL
	       /* the following part initialises the system with two spirals, see PRL 2004, eq 12*/
	       double psi0=Pi; //Vortex
	       // in principle four solutions: +/-( Pi/0 - -0.5*acos(1./xi))
	       //several posibilities
	       //psi0=Pi-0.5*acos(1./xi); // inward spiral
	       
	       ic=0;
	       if(j>Ly/2) jc=5*Ly/8; else {jc=3*Ly/8; psi0=Pi;}
	       kc=Lz/2;
	       distance=sqrt(pow(i-ic,2.0)+pow(j-jc,2.0)+pow(k-kc,2.0));
	       if(distance>0.1){
		 Px[i][j][k]=(i-ic)/distance;
		 Py[i][j][k]=cos(psi0)*(j-jc)/distance-sin(psi0)*(k-kc)/distance;
		 Pz[i][j][k]=sin(psi0)*(j-jc)/distance+cos(psi0)*(k-kc)/distance;

	       }
	       if(distance<2.1){
		 fluid_index[i][j][k]=1;
	       }
	     
#endif


#if ASTER
	       /* the following part initialises the system with an aster*/
	       ic=0;
	       jc=Ly/2;
	       kc=Lz/2;
	       distance=sqrt(pow(i-ic,2.0)+pow(j-jc,2.0)+pow(k-kc,2.0));
	       if(distance>0.1){
		 Px[i][j][k]=(i-ic)/distance;
		 Py[i][j][k]=(j-jc)/distance;
		 Pz[i][j][k]=(k-kc)/distance;
	       }
	       if(distance<2.1){
		 fluid_index[i][j][k]=1;
	       }
	     
#endif

#ifdef NOISY
	       phase=NOISY*2.*Pi*(0.5-drand48());
	       double temp=cos(phase)*Py[i][j][k]+sin(phase)*Pz[i][j][k];
	       Pz[i][j][k]=sin(phase)*Py[i][j][k]+cos(phase)*Pz[i][j][k];
	       Py[i][j][k]=temp;

#endif

	for (l=0; l<15; l++) {
	  f[i][j][k][l]=density[i][j][k]/15.0;
	}
      }//end k loop
    }
  }

}//end initializeP
//-------------------------------------------------------
//--------------Write data------------------------------- 

void writedata(int n){
  ofstream data("field.dat",ios::out|ios::app);
  while (!data.good()){
  data.clear();
  //sleep(10);
  data.open("field.dat",ios::out|ios::app);
  cerr << "writing to datname.c_str(),ios::out|ios::app faild" << endl;
  if (data.good()) cerr << " writing works" << endl;
  }
   for(int i=0; i<Lx; i++) {
      for (int j=0; j<Ly; j++) {
        for (int k=0; k<Lz; k++) {
	  data<< setiosflags(ios::fixed) << setprecision(2) <<	  n<<" "<<i << " "<< j << " "<< k << " "<< Px[i][j][k] << " "<< Py[i][j][k] << " "<< Pz[i][j][k] <<endl;// << " "<< u[i][j][k][0] << " "<< u[i][j][k][1] << " "<< u[i][j][k][2] << endl;
	}//zloop
      }//ylooop
   }//xloop
	data<<endl<<endl;
}
//-------------------------------------------------------



void streamfile(void)
{
  int i,j,k;
  double sch;
  double eig1,eig2,duydz,stress,visc;
  double Qxxl,Qxyl,Qyyl,Qxzl,Qyzl,Qsqxx,Qsqxy,Qsqyy,Qsqzz,Qsqxz,Qsqyz,TrQ2;
  double Hxx,Hxy,Hyy,Hxz,Hyz,sigxx,sigxy,sigyy,sigxz,sigyz,mDQ4yz,DQpQDyz;
  double dQxxdx,dQxxdy,dQxydx,dQxydy,dQyydx,dQyydy,dQxzdx,dQyzdx,dQxzdy,dQyzdy;

  int nrots,emax,enxt;
  double m[3][3],d[3],v[3][3];

if((output = fopen("out.dat","w"))==NULL)
{
 printf("non riesco ad aprire file di output\n");
 exit(0);
}

   for(i=0; i<Lx; i++) {
      for (j=0; j<Ly; j++) {
        for (k=0; k<Lz; k++) {

      fprintf(output, "%d %d %d %16.8f %16.8f %16.8f %16.8f %16.8f %16.8f %16.8f %16.8f %16.8f %16.8f  %16.8f %16.8f  \n",
	i,j,k,Px[i][j][k],Py[i][j][k],Pz[i][j][k],u[i][j][k][0],u[i][j][k][1],u[i][j][k][2],Stressyy[i][j][k],Stressyz[i][j][k],Stresszz[i][j][k],average_shear_stress/Lz,h[i][j][k][0],h[i][j][k][1],h[i][j][k][2]);

       }
     }
  }
  fclose(output);



  i=Lx/2;
  j=Ly/2;
  k=Lz/2;

  fprintf(stderr,"%16.3f %16.3f %16.3f \n",Px[i][j][k],Py[i][j][k],Pz[i][j][k]);

  if ((time(0)-runtime)>started){ 
    cout << (time(0)-started) << endl;
    exit(1);
  }

}

/**************************************************************************/
/* The following routines are based on those given in Numerical Recipes   */
/**************************************************************************/

double gasdev(int start)
{
  static int iset=0;
  static double gset;
  double fac,rsq,v1,v2;

  if (start==1) {
    iset=0;
  }
  if  (iset == 0) {
    do {
      v1=2.0*drand48()-1.0;
      v2=2.0*drand48()-1.0;  
      rsq=v1*v1+v2*v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac=sqrt(-2.0*log(rsq)/rsq);
    gset=v1*fac;
    iset=1;
    return v2*fac;
  } else {
    iset=0;
    return gset;
  }
}





 
  
