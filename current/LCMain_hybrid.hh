
#ifdef PARALLEL
#include <mpi.h>
#endif

// ----------------------------------------
// useful constants

#define Pi 3.141592653589793
#define TwoPi 6.283185307179586
#define sqr2 1.4142136
#define FourPi 12.566370614359173
#define Inv12Pi 0.026525823848649223
#define Inv4Pi 0.079577471545947668
#define Inv8Pi 0.039788735772973834

#ifndef LCMAIN_HH
#define LCMAIN_HH

// ----------------------------------------
// pameters

int Lx,Lx2,Ly,Ly2,Lz,Lz2,Nmax,GRAPHICS,stepskip,itimprov;
int ix1,ix2,jy1,jy2,kz1,kz2;
int FePrintInt, SigPrintFac, OVDPrintInt;

float temperature=0.5;
float Abulk=1.0;
float L1init,L1;
float L2init,L2;

double q0init,q0;
double rr,rr_old;

int numuc;
double numhftwist;
double threshold;

float bodyforce;

float Gamma=0.33775;   // diffusion constant
float gam=3.0;       // 2.701  gamma>2.7 => nematic
double aa=4.0;
double phivr=0.0;      // coupling of hydrodynamics to order parameter
float beta=0.0;        // 2.1349     viscous polymer stress coefficient O(q^2)
float beta2=0.0;       // 0.15159  viscous polymer stress coefficient O(q)
float xi=1.0;          // related to effective aspect ratio, should be <= one
float tau1=0.56334;    // proportional to viscosity
float tau2=0.25;       // magnitude of flow-induced diffusion
float dt=1.0;          //
float densityinit=2.0; // initial value for the density everywhere

// -------------------------------------
// activity
double zeta;
// --------------------------------------
// noise
double noise_strength; // noise strength
int NOISE,seed;

// ----------------------------------------
// electric field
double delVz=2.9784;   // (voltage at z=Lz) - (voltage at z=0)
float epsa=41.4;       // dielectric anisotropy
float epsav=9.8;       // average dielectric constant

// ----------------------------------------
// flexoelectricity
float ef0=0.0;         // not implemented yet
                       // flexoelectric coefficient, expect ef0 << kappa

// ----------------------------------------
// boundary conditions--still need a little work
// #define vwtp 0.0   wall velocities y direction
// #define vwbt -0.0

double vwtp=0.0;
double vwbt=-0.0;

// ----------------------------------------
// Inital configuration 

int RANDOM,TWIST,O2STRUCT,O5STRUCT,O8STRUCT,O8MSTRUCT,DTSTRUCT,HEX3DA,HEX3DB,HEXPLANAR,BLUEHAAR;
int REDSHIFT,BACKFLOW;
 
float bcstren=10.0;  // amplitude of harmonic boundary potential
double wallamp;

float angztop=0.0; /*88.0 0.0*/
float angzbot=0.0; /*92.0*/

float angxytop=0.0;
float angxybot=0.0;
double Qxxtop,Qxytop,Qyytop,Qxztop,Qyztop,Qxxbot,Qxybot,Qyybot,Qxzbot,Qyzbot;
double kappa,tauc,caz;

// ============================================================
// function declarations

void parametercalc(int);
void equilibriumdist(void);
void initialize(void);
void reinit();
void randomizeQ(void);
void startDroplet(void);
void startSlab(void);
void collisionop(void);
void update0(double****, double****);
void update(double****, double****);
void streamout(void);
void streamfile(const int iter);
double pickphase(double, double);
void writeStartupFile();
void setboundary();
void readStartupFile(const int);
void writeDiscFile(const int iter);
void computeStressFreeEnergy(const int iter);

// modified numerical recipes routines
double gasdev(int);
void fourn(double [], unsigned long [], int, int);
void rlft3(double ***, double **, unsigned long, unsigned long,
	unsigned  long, int);
void nrerror(char error_text[]);
double **dmatrix(long nrl, long nrh, long ncl, long nch);
double ***d3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh);
void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch);
void free_d3tensor(double ***t, long nrl, long nrh, long ncl, long nch,
	long ndl, long ndh);

void jacobi(double (*a)[3], double d[], double (*v)[3], int *nrot);


// global variables


double ****fa,****gaxx,****gaxy;
double ****gayy,****gaxz,****gayz;
double ****fb,****gbxx,****gbxy;
double ****gbyy,****gbxz,****gbyz;
double ****Fc,****Gxxc,****Gxyc;
double ****Gyyc,****Gxzc,****Gyzc;

double freeenergy,freeenergytwist,txx,txy,txz,tyz,tyx,tyy,tzx,tzy,tzz;
double lastFreeenergy;
int numCase;

double ****feq,****gxxeq,****gxyeq;
double ****gyyeq,****gxzeq,****gyzeq;

double ***density,***Qxx,***Qxy,***Qyy;
double ***Qxz,***Qyz,****u,****Fh;
double ***Qxxold,***Qxyold,***Qyyold;
double ***Qxzold,***Qyzold;
double ***Qxxnew,***Qxynew,***Qyynew;
double ***Qxznew,***Qyznew; 
double ***Qxxinit,***Qxyinit,***Qyyinit;
double ***Qxzinit,***Qyzinit;
double ***DEHxx,***DEHxy,***DEHyy;
double ***DEHxz,***DEHyz;
double ***DEHxxold,***DEHxyold,***DEHyyold;
double ***DEHxzold,***DEHyzold;
double ***DEH1xx,***DEH1xy,***DEH1yy;
double ***DEH1xz,***DEH1yz;
double ***DEH2xx,***DEH2xy,***DEH2yy;
double ***DEH2xz,***DEH2yz;
double ***DEH3xx,***DEH3xy,***DEH3yy;
double ***DEH3xz,***DEH3yz;
double ***molfieldxx, ***molfieldxy,***molfieldxz;
double ***molfieldyy, ***molfieldyz;
double ***DG2xx,***DG2xy,***DG2yy;
double ***DG2xz,***DG2yz,***DG2zz;
double ***DG2yx,***DG2zx,***DG2zy;
double ***Stressxx,***Stressxy,***Stressyy;
double ***Stressxz,***Stressyz,***Stresszz;
double ***Stressyx,***Stresszx,***Stresszy;
double ***tauxy,***tauxz,***tauyz;
double ***Ex,***Ey,***Ez;
double ***Pdx,***Pdy,***Pdz;

double ****f,****gxx,****gxy;
double ****gyy,****gxz,****gyz;
double ****fpr,****gxxpr,****gxypr;
double ****gyypr,****gxzpr,****gyzpr;

double oneplusdtover2tau1,oneplusdtover2tau2,qvisc; 
int active,poiseuille;
int e[15][3];

ofstream output;
ofstream output1;
ofstream fp;

int myPE,nbPE,nbPErow;

// Appropriate for serial version

int pe_cartesian_size_[3];
int pe_cartesian_coordinates_[3];
int io_ngroups_;
int io_group_id_;

#ifdef PARALLEL

MPI_Status status;

int leftNeighbor, rightNeighbor,upNeighbor,downNeighbor;

/* KS timers */
double total_exch_ = 0.0;
double total_comm_ = 0.0;
double total_io_   = 0.0;
double total_time_ = 0.0;

MPI_Comm cartesian_communicator_;
int pe_cartesian_neighbour_[2][3];

MPI_Comm io_communicator_;
int io_group_size_;
int io_rank_;

#endif /* PARALLEL */

#endif
