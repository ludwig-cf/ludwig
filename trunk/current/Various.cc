/****************************************************************************
 *
 *  3D streamfile I/O outputs: stress, velocity, and molfield
 *
 *  Each I/O communicator opens its own file and writes the
 *  appropriate local lattice information as output.
 *
 *  Within each communicator, processes block until they receive
 *  the token to procede.
 *
 ****************************************************************************/

void streamfile_ks(const int iter) {

  int i,j,k;  
  int nrots,emax,enxt;
  double m[3][3],d[3],v[3][3];
  int ioff = 0, joff = 0, koff = 0;

#ifdef _BINARY_IO_
  int ibuf[3];
  double rbuf[10];
#endif

  /* Build the filename for this communicator stub.case.iteration.dat-id-n */

  String oname("order_velo.");
  oname.concat(numCase);
  oname.concat(".");
  oname.concat(iter);
  oname.concat(".dat-");
  oname.concat(io_group_id_);
  oname.concat("-");
  oname.concat(io_ngroups_);

  String sname("dir.");
  sname.concat(numCase);
  sname.concat(".");
  sname.concat(iter);
  sname.concat(".dat-");
  sname.concat(io_group_id_);
  sname.concat("-");
  sname.concat(io_ngroups_);


#ifdef PARALLEL
  int token=0;
  const int tag = 986;
  double t0, t1;

  ioff = Lx*pe_cartesian_coordinates_[0]/pe_cartesian_size_[0];
  joff = Ly*pe_cartesian_coordinates_[1]/pe_cartesian_size_[1];
  koff = Lz*pe_cartesian_coordinates_[2]/pe_cartesian_size_[2];

  t0 = MPI_Wtime();

  if (io_rank_ == 0) {
    output.open(oname.get());
    output1.open(sname.get());
  }
  else {
    MPI_Recv(&token, 1, MPI_INT, io_rank_ - 1, tag, io_communicator_,
	     &status);
    output.open(oname.get(),ios::app);
    output1.open(sname.get(),ios::app);
  }
#else
  // Serial
  output.open(oname.get());
  output1.open(sname.get());
#endif

  output.precision(5);
       
  for(i=ix1; i<ix2; i++) { 
    for (j=jy1; j<jy2; j++) {
      for (k=kz1; k<kz2; k++) {
    
      m[0][0]=Qxx[i][j][k];
      m[0][1]=Qxy[i][j][k];
      m[0][2]=Qxz[i][j][k];
      m[1][0]=Qxy[i][j][k];
      m[1][1]=Qyy[i][j][k];
      m[1][2]=Qyz[i][j][k];
      m[2][0]=Qxz[i][j][k];
      m[2][1]=Qyz[i][j][k];
      m[2][2]= -(m[0][0]+m[1][1]);

      jacobi(m,d,v,&nrots);

      if (d[0] > d[1]) {
	emax=0;
	enxt=1;
      }
      else {
	emax=1;
	enxt=0;
      }
      if (d[2] > d[emax]) {
	emax=2;
      }
      else if (d[2] > d[enxt]) {
	enxt=2;
      }

#ifdef _BINARY_IO_
      ibuf[0] = i-ix1+ioff;
      ibuf[1] = j-jy1+joff;
      ibuf[2] = k-kz1+koff;
      rbuf[0] = Qxx[i][j][k];
      rbuf[1] = Qxy[i][j][k];
      rbuf[2] = Qxz[i][j][k];
      rbuf[3] = Qyy[i][j][k];
      rbuf[4] = Qyz[i][j][k];
      rbuf[5] = u[i][j][k][0];
      rbuf[6] = u[i][j][k][1];
      rbuf[7] = u[i][j][k][2];
      rbuf[8] = d[emax];
      rbuf[9] = molfieldxx[i][j][k];

      output.write((char *) ibuf, 3*sizeof(int));
      output.write((char *) rbuf, 10*sizeof(double));

      rbuf[0] = v[0][emax];
      rbuf[1] = v[1][emax];
      rbuf[2] = v[2][emax];

      output1.write((char *) ibuf, 3*sizeof(int));
      output1.write((char *) rbuf, 3*sizeof(double));
#else
      /* Usual ascii form */
      output << i-ix1+ioff << " " << j-jy1+joff << " " << k-kz1+koff << " " 
	     << Qxx[i][j][k] << " " << Qxy[i][j][k] << " " 
	     << Qxz[i][j][k]<< " " << Qyy[i][j][k]<< " " 
	     << Qyz[i][j][k]<< " " << u[i][j][k][0]<< " " 
	     << u[i][j][k][1]<< " " << u[i][j][k][2] << " "
 	     << d[emax]<< " " << molfieldxx[i][j][k] << endl;

      output1 << i-ix1+ioff << " " << j-jy1+joff << " " << k-kz1+koff
	      << " " << v[0][emax]
	      << " " << v[1][emax]<< " " << v[2][emax] << endl;
#endif
       } 

    }
  }
  output.close();
  output1.close();

#ifdef PARALLEL
  if (io_rank_ < io_group_size_ - 1) {
    MPI_Send(&token, 1, MPI_INT, io_rank_ + 1, tag, io_communicator_);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  total_io_ += (t1-t0);
#endif

  return;
}


void writeDiscFile_ks(const int iter) {

  int i,j,k;
  int nrots,emax,enxt;
  double m[3][3],d[3],v[3][3];
  int ioff = 0, joff = 0, koff = 0;   /* lattice offsets */
  int ic, jc, kc;                     /* global lattice positions */

#ifdef _BINARY_IO_
  int ibuf[3];
#endif

  String oname("disc.");
  oname.concat(numCase);
  oname.concat(".");
  oname.concat(iter);
  oname.concat(".dat-");
  oname.concat(io_group_id_);
  oname.concat("-");
  oname.concat(io_ngroups_);

#ifdef PARALLEL
  int token=0;
  const int tag = 987;
  double t0, t1;

  ioff = Lx*pe_cartesian_coordinates_[0]/pe_cartesian_size_[0];
  joff = Ly*pe_cartesian_coordinates_[1]/pe_cartesian_size_[1];
  koff = Lz*pe_cartesian_coordinates_[2]/pe_cartesian_size_[2];

  t0 = MPI_Wtime();

  if (io_rank_ == 0)
    output.open(oname.get());
  else {
    MPI_Recv(&token, 1, MPI_INT, io_rank_ - 1, tag, io_communicator_,
	     &status);
    output.open(oname.get(),ios::app);
  }
#else
    output.open(oname.get());
#endif

  output.precision(4);
       
  for(i=ix1; i<ix2; i++) { 
    for (j=jy1; j<jy2; j++) {
      for (k=kz1; k<kz2; k++) {
    
	ic = i - ix1 + ioff;
	jc = j - jy1 + joff;
	kc = k - kz1 + koff;

      m[0][0]=Qxx[i][j][k];
      m[0][1]=Qxy[i][j][k];
      m[0][2]=Qxz[i][j][k];
      m[1][0]=Qxy[i][j][k];
      m[1][1]=Qyy[i][j][k];
      m[1][2]=Qyz[i][j][k];
      m[2][0]=Qxz[i][j][k];
      m[2][1]=Qyz[i][j][k];
      m[2][2]= -(m[0][0]+m[1][1]);
      jacobi(m,d,v,&nrots);

      if (d[0] > d[1]) {
	emax=0;
	enxt=1;
      }
      else {
	emax=1;
	enxt=0;
      }
      if (d[2] > d[emax]) {
	emax=2;
      }
      else if (d[2] > d[enxt]) {
	enxt=2;
      }

      if (d[emax] < threshold) {
#ifdef _BINARY_IO_
	ibuf[0] = ic;
	ibuf[1] = jc;
	ibuf[2] = kc;
	output.write((char *) ibuf, 3*sizeof(int));
#else
	output << ic << " " << jc << " " << kc << endl;
#endif
      }
      } 
    }
  }
  output.close();

#ifdef PARALLEL
  if (io_rank_ < io_group_size_ - 1) {
    MPI_Send(&token, 1, MPI_INT, io_rank_ + 1, tag, io_communicator_);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();

  total_io_ += (t1-t0);
#endif

  return;
}

/*****************************************************************************
 *
 *  writeRestart
 *
 *  Write out the entire state (distributions f, five components
 *  of Q tensor) to file, suitable to reading in again for a
 *  restart.
 *
 *  The current value of the redshift 'rr' is also required.
 *
 *  The argument is the time step.
 *
 *  This is always binary.
 *
 *****************************************************************************/

void writeRestart(const int iter) {

  int i, j, k, p;  
  double buffer[20];

  /* Build the filename for this communicator stub.case.iteration.dat-id-n */

  String oname("config.");
  oname.concat(numCase);
  oname.concat(".");
  oname.concat(iter);
  oname.concat(".");
  oname.concat(io_group_id_);
  oname.concat("-");
  oname.concat(io_ngroups_);

#ifdef PARALLEL
  int token=0;
  const int tag = 989;
  double t0, t1;

  t0 = MPI_Wtime();

  if (io_rank_ == 0) {
    output.open(oname.get());
  }
  else {
    MPI_Recv(&token, 1, MPI_INT, io_rank_ - 1, tag, io_communicator_,
	     &status);
    output.open(oname.get(),ios::app);
  }
#else
  // Serial
  output.open(oname.get());
#endif
       
  for(i=ix1; i<ix2; i++) { 
    for (j=jy1; j<jy2; j++) {
      for (k=kz1; k<kz2; k++) {
	
	for (p = 0; p < 15; p++) {
	  buffer[p] = f[i][j][k][p];
	}
	buffer[15] = Qxx[i][j][k];
	buffer[16] = Qxy[i][j][k];
	buffer[17] = Qxz[i][j][k];
	buffer[18] = Qyy[i][j][k];
	buffer[19] = Qyz[i][j][k];
	output.write((char *) buffer, 20*sizeof(double));
      }
    }
  }

  // variable and fixed (BC) redshift
  buffer[0] = rr;
  buffer[1] = rr_fxd;
  output.write((char *) buffer, 2*sizeof(double));

  // fixed Q along decomposition rows for wall interpolation
  for(i=ix1; i<ix2; i++) { 
    for (j=1; j<=Ly; j++) {
      for (k=kz1; k<kz2; k++) {
	buffer[0] = Qxxfxd_all_y[i][j][k];
	buffer[1] = Qxyfxd_all_y[i][j][k];
	buffer[2] = Qxzfxd_all_y[i][j][k];
	buffer[3] = Qyyfxd_all_y[i][j][k];
	buffer[4] = Qyzfxd_all_y[i][j][k];
	output.write((char *) buffer, 5*sizeof(double));
      }
    }
  }

  output.close();

#ifdef PARALLEL
  if (io_rank_ < io_group_size_ - 1) {
    MPI_Send(&token, 1, MPI_INT, io_rank_ + 1, tag, io_communicator_);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();

  total_io_ += (t1-t0);
#endif

  return;
}

/*****************************************************************************
 *
 *  readRestart
 *
 *  Read configuration (distributions f, five components of
 *  the Q tensor, a single value of redshift) from file.
 *
 *  The argument is the time step.
 *
 *  Binary is always expected.
 *
 *****************************************************************************/

void readRestart(const int iter) {

  int i, j, k, p;
  double buffer[20];
  ifstream input;

  /* Build the filename for this communicator stub.case.iteration.dat-id-n */

  String iname("config.");
  iname.concat(numCase);
  iname.concat(".");
  iname.concat(iter);
  iname.concat(".");
  iname.concat(io_group_id_);
  iname.concat("-");
  iname.concat(io_ngroups_);
  int token=0;

#ifdef PARALLEL
  const int tag = 990;
  double t0, t1;

  t0 = MPI_Wtime();

  if (io_rank_ == 0) {
    input.open(iname.get());
  }
  else {
    /* Block until we get the token, which is the position to read from... */
    MPI_Recv(&token, 1, MPI_INT, io_rank_ - 1, tag, io_communicator_,
	     &status);
    input.open(iname.get(),ios::in);
    input.seekg(token, ios::beg);
  }
#else
  // Serial
  input.open(iname.get());
#endif
       
  for(i=ix1; i<ix2; i++) { 
    for (j=jy1; j<jy2; j++) {
      for (k=kz1; k<kz2; k++) {

	input.read((char *) buffer, 20*sizeof(double));
	
	for (p = 0; p < 15; p++) {
	  f[i][j][k][p] = buffer[p];
	}
	Qxx[i][j][k] = buffer[15];
	Qxy[i][j][k] = buffer[16];
	Qxz[i][j][k] = buffer[17];
	Qyy[i][j][k] = buffer[18];
	Qyz[i][j][k] = buffer[19];
      }
    }
  }

  // variable and fixed (BC) redshift
  input.read((char *) buffer, 2*sizeof(double));
  rr = buffer[0];
  rr_fxd = buffer[1];

  // fixed Q along decomposition rows for wall interpolation
  for(i=ix1; i<ix2; i++) { 
    for (j=1; j<=Ly; j++) {
      for (k=kz1; k<kz2; k++) {

	input.read((char *) buffer, 5*sizeof(double));

	Qxxfxd_all_y[i][j][k]=buffer[0];
	Qxyfxd_all_y[i][j][k]=buffer[1];
	Qxzfxd_all_y[i][j][k]=buffer[2];
	Qyyfxd_all_y[i][j][k]=buffer[3];
	Qyzfxd_all_y[i][j][k]=buffer[4];
      }
    }
  }

  if (input.good()) {
    token = input.tellg();
  }
  else {
    cout << "Error on read" << endl;
#ifdef _PARALLEL_
    MPI_Abort(MPI_COMM_WORLD, 0);
#endif
  }

  input.close();

#ifdef PARALLEL
  if (io_rank_ < io_group_size_ - 1) {
    /* Send the token on to the next process */
    MPI_Send(&token, 1, MPI_INT, io_rank_ + 1, tag, io_communicator_);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();

  total_io_ += (t1-t0);
#endif

  return;
}

void get_Qfxd(void){

int ic,jc,kc;

// get Qfxd and redshift
for (ic=ix1; ic<ix2; ic++){
   for (jc=jy1; jc<jy2; jc++){
      for (kc=kz1; kc<kz2; kc++){
	 Qxxfxd[ic][jc][kc]=Qxx[ic][jc][kc];
	 Qxyfxd[ic][jc][kc]=Qxy[ic][jc][kc];
	 Qxzfxd[ic][jc][kc]=Qxz[ic][jc][kc];
	 Qyyfxd[ic][jc][kc]=Qyy[ic][jc][kc];
	 Qyzfxd[ic][jc][kc]=Qyz[ic][jc][kc];
      }
   }
}
rr_fxd=rr;

// send Qfxd along y for interpolation during wall movement
communicateQfxd();

return;
}

void set_Q(int n, int n_flow){

int ic,jc,kc,joff;
extern int kz1,kz2;
double x[Ly+2],y[Ly+2],xi[Ly+2],yi[Ly+2];
double dy_tp,dy_bt;

// set redshift and Q at moving 
// z-boundary via interpolation

rr=rr_fxd;

joff = Ly*pe_cartesian_coordinates_[1]/pe_cartesian_size_[1];

if (pe_cartesian_coordinates_[2] == 0){

   dy_bt=fmod(fabs(vwbt)*(n-n_flow),Ly);

   kc=kz1;
   for (ic=ix1; ic<ix2; ic++){
      for (jc=1; jc<=Ly; jc++){
	 x[jc]=jc;
	 xi[jc]=jc;
	 if(vwbt!=0) xi[jc]=fmod(jc-(vwbt/fabs(vwbt))*dy_bt+Ly-1.,Ly)+1.;
      }

      // Qxx
      for (jc=1; jc<=Ly; jc++){
	 y[jc]=Qxxfxd_all_y[ic][jc][kc];
      }
      cspline(x,y,xi,yi);

      for (jc=jy1; jc<=jy2; jc++){
	 Qxx[ic][jc][kc]=yi[joff+jc];
      }
      // Qxy
      for (jc=1; jc<=Ly; jc++){
	 y[jc]=Qxyfxd_all_y[ic][jc][kc];
      }
      cspline(x,y,xi,yi);
      for (jc=jy1; jc<=jy2; jc++){
	 Qxy[ic][jc][kc]=yi[joff+jc];
      }
      // Qxz
      for (jc=1; jc<=Ly; jc++){
	 y[jc]=Qxzfxd_all_y[ic][jc][kc];
      }
      cspline(x,y,xi,yi);
      for (jc=jy1; jc<=jy2; jc++){
	 Qxz[ic][jc][kc]=yi[joff+jc];
      }
      // Qyy
      for (jc=1; jc<=Ly; jc++){
	 y[jc]=Qyyfxd_all_y[ic][jc][kc];
      }
      cspline(x,y,xi,yi);
      for (jc=jy1; jc<=jy2; jc++){
	 Qyy[ic][jc][kc]=yi[joff+jc];
      }
      // Qyz
      for (jc=1; jc<=Ly; jc++){
	 y[jc]=Qyzfxd_all_y[ic][jc][kc];
      }
      cspline(x,y,xi,yi);
      for (jc=jy1; jc<=jy2; jc++){
	 Qyz[ic][jc][kc]=yi[joff+jc];
      }
   }
}

if (pe_cartesian_coordinates_[2] == pe_cartesian_size_[2]-1){

   dy_tp=fmod(fabs(vwtp)*(n-n_flow),Ly);

   kc=kz2-1;
   for (ic=ix1; ic<ix2; ic++){
      for (jc=1; jc<=Ly; jc++){
	 x[jc]=jc;
	 xi[jc]=jc;
	 if(vwtp!=0) xi[jc]=fmod(jc-(vwtp/fabs(vwtp))*dy_tp+Ly-1.,Ly)+1.;
      }

      // Qxx
      for (jc=1; jc<=Ly; jc++){
	 y[jc]=Qxxfxd_all_y[ic][jc][kc];
      }
      cspline(x,y,xi,yi);

      for (jc=jy1; jc<=jy2; jc++){
	 Qxx[ic][jc][kc]=yi[joff+jc];
      }
      // Qxy
      for (jc=1; jc<=Ly; jc++){
	 y[jc]=Qxyfxd_all_y[ic][jc][kc];
      }
      cspline(x,y,xi,yi);
      for (jc=jy1; jc<=jy2; jc++){
	 Qxy[ic][jc][kc]=yi[joff+jc];
      }
      // Qxz
      for (jc=1; jc<=Ly; jc++){
	 y[jc]=Qxzfxd_all_y[ic][jc][kc];
      }
      cspline(x,y,xi,yi);
      for (jc=jy1; jc<=jy2; jc++){
	 Qxz[ic][jc][kc]=yi[joff+jc];
      }
      // Qyy
      for (jc=1; jc<=Ly; jc++){
	 y[jc]=Qyyfxd_all_y[ic][jc][kc];
      }
      cspline(x,y,xi,yi);
      for (jc=jy1; jc<=jy2; jc++){
	 Qyy[ic][jc][kc]=yi[joff+jc];
      }
      // Qyz
      for (jc=1; jc<=Ly; jc++){
	 y[jc]=Qyzfxd_all_y[ic][jc][kc];
      }
      cspline(x,y,xi,yi);
      for (jc=jy1; jc<=jy2; jc++){
	 Qyz[ic][jc][kc]=yi[joff+jc];
      }
   }
}

return;
}


void cspline(double x[],double y[],double xi[],double yi[]){

int a;
double xs[Ly+1],ys[Ly+1],xt[Ly+2],yt[Ly+2];
double yp0,ypnp1,y2[Ly+3],yy[1];

for (a=0;a<=Ly-1;a++){ 
   xs[a]=x[a+1];
   ys[a]=y[a+1];
}

xs[Ly]=xs[0]+Ly;
ys[Ly]=ys[0];

for (a=1;a<=Ly;a++){ 
   xt[a]=x[a];
   yt[a]=y[a];
}

xt[0]=0;
yt[0]=yt[Ly];

xt[Ly+1]=Ly+1;
yt[Ly+1]=yt[1];

yp0=(yt[1]-yt[0])/(xt[1]-xt[0]);
ypnp1=(yt[Ly+1]-yt[Ly])/(xt[Ly+1]-xt[Ly]);

spline(xt,yt,Ly+2,yp0,ypnp1,y2);

for(a=1;a<=Ly;a++){
   splint(xt,yt,y2,Ly+2,xi[a],yy);
   yi[a]=yy[0];
}

yi[0]=yi[Ly];
yi[Ly+1]=yi[1];

}


void spline(double x[],double y[],int n,double yp1,double ypn,double y2[]){

        int i,k;
        double p,qn,sig,un,u[n];

        if (yp1 > 0.99e30)
                y2[1]=u[1]=0.0;
        else {
                y2[1] = -0.5;
                u[1]=(3.0/(x[2]-x[1]))*((y[2]-y[1])/(x[2]-x[1])-yp1);
        }
        for (i=2;i<=n-1;i++) {
                sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
                p=sig*y2[i-1]+2.0;
                y2[i]=(sig-1.0)/p;
                u[i]=(y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]);
                u[i]=(6.0*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;
        }
        if (ypn > 0.99e30)
                qn=un=0.0;
        else {
                qn=0.5;
                un=(3.0/(x[n]-x[n-1]))*(ypn-(y[n]-y[n-1])/(x[n]-x[n-1]));
        }
        y2[n]=(un-qn*u[n-1])/(qn*y2[n-1]+1.0);
        for (k=n-1;k>=1;k--)
                y2[k]=y2[k]*y2[k+1]+u[k];
}

void splint(double xa[],double ya[],double y2a[],int n,double x,double y[]){

        int klo,khi,k;
        double h,b,a;

        klo=1;
        khi=n;
        while (khi-klo > 1) {
                k=(khi+klo) >> 1;
                if (xa[k] > x) khi=k;
                else klo=k;
        }
        h=xa[khi]-xa[klo];
        if (h == 0.0) {
	    printf("Bad XA input to routine SPLINT");
	    exit(0);
	}
        a=(xa[khi]-x)/h;
        b=(x-xa[klo])/h;

        y[0]=a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;
}

