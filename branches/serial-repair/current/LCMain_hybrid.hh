

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
double namp=0.0;       // noise amplitude

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

int RANDOM,TWIST,O2STRUCT,O5STRUCT,O8STRUCT,O8MSTRUCT,DTSTRUCT,HEX3D,HEXPLANAR,BLUEHAAR;
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
#define n 3
void jacobi(double (*a)[n], double d[], double (*v)[n], int *nrot);
#undef n

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

// double fa[Lx][Ly][Lz][15],gaxx[Lx][Ly][Lz][15],gaxy[Lx][Ly][Lz][15];
// double gayy[Lx][Ly][Lz][15],gaxz[Lx][Ly][Lz][15],gayz[Lx][Ly][Lz][15];
// double fb[Lx][Ly][Lz][15],gbxx[Lx][Ly][Lz][15],gbxy[Lx][Ly][Lz][15];
// double gbyy[Lx][Ly][Lz][15],gbxz[Lx][Ly][Lz][15],gbyz[Lx][Ly][Lz][15];
// double Fc[Lx][Ly][Lz][15],Gxxc[Lx][Ly][Lz][15],Gxyc[Lx][Ly][Lz][15];
// double Gyyc[Lx][Ly][Lz][15],Gxzc[Lx][Ly][Lz][15],Gyzc[Lx][Ly][Lz][15];
// double freeenergy,freeenergytwist,txx,txy,txz,tyz,tyx,tyy,tzx,tzy,tzz;
// double feq[Lx][Ly][Lz][15],gxxeq[Lx][Ly][Lz][15],gxyeq[Lx][Ly][Lz][15];
// double gyyeq[Lx][Ly][Lz][15],gxzeq[Lx][Ly][Lz][15],gyzeq[Lx][Ly][Lz][15];
// double density[Lx][Ly][Lz],Qxx[Lx][Ly][Lz],Qxy[Lx][Ly][Lz],Qyy[Lx][Ly][Lz];
// double Qxz[Lx][Ly][Lz],Qyz[Lx][Ly][Lz],u[Lx][Ly][Lz][3];
// double Qxxinit[Lx][Ly][Lz],Qxyinit[Lx][Ly][Lz],Qyyinit[Lx][Ly][Lz];
// double Qxzinit[Lx][Ly][Lz],Qyzinit[Lx][Ly][Lz];
// double DEHxx[Lx][Ly][Lz],DEHxy[Lx][Ly][Lz],DEHyy[Lx][Ly][Lz];
// double DEHxz[Lx][Ly][Lz],DEHyz[Lx][Ly][Lz];
// double DEH1xx[Lx][Ly][Lz],DEH1xy[Lx][Ly][Lz],DEH1yy[Lx][Ly][Lz];
// double DEH1xz[Lx][Ly][Lz],DEH1yz[Lx][Ly][Lz];
// double DEH2xx[Lx][Ly][Lz],DEH2xy[Lx][Ly][Lz],DEH2yy[Lx][Ly][Lz];
// double DEH2xz[Lx][Ly][Lz],DEH2yz[Lx][Ly][Lz];
// double DEH3xx[Lx][Ly][Lz],DEH3xy[Lx][Ly][Lz],DEH3yy[Lx][Ly][Lz];
// double DEH3xz[Lx][Ly][Lz],DEH3yz[Lx][Ly][Lz];
// double DG2xx[Lx][Ly][Lz],DG2yy[Lx][Ly][Lz],DG2xy[Lx][Ly][Lz];
// double DG2xz[Lx][Ly][Lz],DG2yz[Lx][Ly][Lz],DG2zz[Lx][Ly][Lz],pG[Lx][Ly][Lz];
// double tauxy[Lx][Ly][Lz],tauxz[Lx][Ly][Lz],tauyz[Lx][Ly][Lz];
// double Ex[Lx][Ly][Lz],Ey[Lx][Ly][Lz],Ez[Lx][Ly][Lz];
// double Pdx[Lx][Ly][Lz],Pdy[Lx][Ly][Lz],Pdz[Lx][Ly][Lz];

double ****f,****gxx,****gxy;
double ****gyy,****gxz,****gyz;
double ****fpr,****gxxpr,****gxypr;
double ****gyypr,****gxzpr,****gyzpr;

double oneplusdtover2tau1,oneplusdtover2tau2,qvisc; 
int pouiseuille;
int pouiseuille1;
int e[15][3];

ofstream output;
ofstream output1;
ofstream fp;

int myPE,nbPE,nbPErow;

#ifdef PARALLEL

MPI_Status status;
//MPI_Request req[10];
//MPI_Request reqBsend;

//MPI_Datatype leftFieldsType,rightFieldsType;

int leftNeighbor, rightNeighbor,upNeighbor,downNeighbor;

/* KS timers */
double total_exch_ = 0.0;
double total_comm_ = 0.0;
double total_io_   = 0.0;
double total_time_ = 0.0;

MPI_Comm cartesian_communicator_;
int pe_cartesian_size_[3];
int pe_cartesian_coordinates_[3];
int pe_cartesian_neighbour_[2][3];

MPI_Comm io_communicator_;
int io_ngroups_ = 1; /* Default to 1 file for output */
int io_group_id_;
int io_group_size_;
int io_rank_;

#endif

#else

// ----------------------------------------
// pameters

extern int FePrintInt;
extern int Lx,Lx2,Ly,Ly2,Lz,Lz2,Nmax,GRAPHICS,stepskip,itimprov;
extern int ix1,ix2,jy1,jy2,kz1,kz2;

extern float temperature;
extern float Abulk;
extern float L1init,L1;
extern float L2init,L2;

extern double q0init,q0;
extern double rr,rr_old;

extern int numuc;
extern double numhftwist;
extern double threshold;

extern float bodyforce;

extern float Gamma;           // diffusion constant
extern float gam;             // 2.701  gamma>2.7 => nematic
extern double aa;
extern double phivr;      // coupling of hydrodynamics to order parameter
extern float beta;        // 2.1349     viscous polymer stress coefficient O(q^2)
extern float beta2;       // 0.15159  viscous polymer stress coefficient O(q)
extern float xi;          // related to effective aspect ratio, should be <= one
extern float tau1;    // proportional to viscosity
extern float tau2;       // magnitude of flow-induced diffusion
extern float dt;          //
extern float densityinit; // initial value for the density everywhere
extern double namp;       // noise amplitude

// ----------------------------------------
// electric field
extern double delVz;   // (voltage at z=Lz) - (voltage at z=0)
extern float epsa;       // dielectric anisotropy
extern float epsav;       // average dielectric constant

// ----------------------------------------
// flexoelectricity
extern float ef0;         // not implemented yet
                       // flexoelectric coefficient, expect ef0 << kappa

// ----------------------------------------
// boundary conditions--still need a little work
// #define vwtp 0.0   wall velocities y direction
// #define vwbt -0.0

extern double vwtp;
extern double vwbt;

// ----------------------------------------
// Inital configuration 

extern int RANDOM,TWIST,O2STRUCT,O5STRUCT,O8STRUCT,O8MSTRUCT,DTSTRUCT,HEX3D,HEXPLANAR,BLUEHAAR;
extern int REDSHIFT,BACKFLOW; 

extern float bcstren;  // amplitude of harmonic boundary potential
extern double wallamp;

extern float angztop; /*88.0 0.0*/
extern float angzbot; /*92.0*/

extern float angxytop;
extern float angxybot;
extern double Qxxtop,Qxytop,Qyytop,Qxztop,Qyztop,
       Qxxbot,Qxybot,Qyybot,Qxzbot,Qyzbot;
extern double kappa,tauc,caz;

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
void streamfile(void);
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
#define n 3
void jacobi(double (*a)[n], double d[], double (*v)[n], int *nrot);
#undef n

// global variables


extern double ****fa,****gaxx,****gaxy;
extern double ****gayy,****gaxz,****gayz;
extern double ****fb,****gbxx,****gbxy;
extern double ****gbyy,****gbxz,****gbyz;
extern double ****Fc,****Gxxc,****Gxyc;
extern double ****Gyyc,****Gxzc,****Gyzc;

extern double freeenergy,freeenergytwist,txx,txy,txz,tyz,tyx,tyy,tzx,tzy,tzz;
extern double lastFreeenergy;
extern int numCase;

extern double ****feq,****gxxeq,****gxyeq;
extern double ****gyyeq,****gxzeq,****gyzeq;

extern double ***density,***Qxx,***Qxy,***Qyy;
extern double ***Qxz,***Qyz,****u,****Fh;
extern double ***Qxxold,***Qxyold,***Qyyold;
extern double ***Qxzold,***Qyzold;
extern double ***Qxxnew,***Qxynew,***Qyynew;
extern double ***Qxznew,***Qyznew; 
extern double ***Qxxinit,***Qxyinit,***Qyyinit;
extern double ***Qxzinit,***Qyzinit;
extern double ***DEHxx,***DEHxy,***DEHyy;
extern double ***DEHxz,***DEHyz;
extern double ***DEHxxold,***DEHxyold,***DEHyyold;
extern double ***DEHxzold,***DEHyzold;
extern double ***DEH1xx,***DEH1xy,***DEH1yy;
extern double ***DEH1xz,***DEH1yz;
extern double ***DEH2xx,***DEH2xy,***DEH2yy;
extern double ***DEH2xz,***DEH2yz;
extern double ***DEH3xx,***DEH3xy,***DEH3yy;
extern double ***DEH3xz,***DEH3yz;
extern double ***molfieldxx, ***molfieldxy,***molfieldxz;
extern double ***molfieldyy, ***molfieldyz;
extern double ***DG2xx,***DG2xy,***DG2yy;
extern double ***DG2xz,***DG2yz,***DG2zz;
extern double ***DG2yx,***DG2zx,***DG2zy;
extern double ***Stressxx,***Stressxy,***Stressyy;
extern double ***Stressxz,***Stressyz,***Stresszz;
extern double ***Stressyx,***Stresszx,***Stresszy;
extern double ***tauxy,***tauxz,***tauyz;
extern double ***Ex,***Ey,***Ez;
extern double ***Pdx,***Pdy,***Pdz;

extern double ****f,****gxx,****gxy;
extern double ****gyy,****gxz,****gyz;
extern double ****fpr,****gxxpr,****gxypr;
extern double ****gyypr,****gxzpr,****gyzpr;

extern double oneplusdtover2tau1,oneplusdtover2tau2,qvisc; 
extern int pouiseuille;
extern int pouiseuille1;
extern int e[15][3];

extern ofstream output;
extern ofstream output1;
extern ofstream fp;

extern int myPE,nbPE,nbPErow;

#ifdef PARALLEL

extern MPI_Status status;
extern MPI_Request req[10];
extern MPI_Request reqBsend;

extern MPI_Datatype leftFieldsType,rightFieldsType;

extern int leftNeighbor, rightNeighbor,upNeighbor,downNeighbor;


extern double *tmpBuf;
#endif

#endif
