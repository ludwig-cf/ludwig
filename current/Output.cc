void computeStressFreeEnergy(int n)
{
  int i,j,k,p,iup,idwn,jup,jdwn,kup,kdwn;
  double dQxxdx,dQxxdy,dQxxdz,dQxydx,dQxydy,dQxydz,dQyydx,dQyydy,dQyydz;
  double dQxzdx,dQxzdy,dQxzdz,dQyzdx,dQyzdy,dQyzdz;
  double dQyxdx,dQyxdy,dQyxdz;
  double dQzxdx,dQzxdy,dQzxdz,dQzydx,dQzydy,dQzydz;
  double dQzzdx,dQzzdy,dQzzdz;
  double trt,trd2Q,TrQ2;
  double d2Qxxdxdx,d2Qxxdydy,d2Qxxdxdy,d2Qxxdzdz,d2Qxxdxdz,d2Qxxdydz;
  double d2Qyydxdx,d2Qyydydy,d2Qyydxdy,d2Qyydzdz,d2Qyydxdz,d2Qyydydz;
  double d2Qxydxdx,d2Qxydydy,d2Qxydxdy,d2Qxydzdz,d2Qxydxdz,d2Qxydydz;
  double d2Qxzdxdx,d2Qxzdydy,d2Qxzdxdy,d2Qxzdzdz,d2Qxzdxdz,d2Qxzdydz;
  double d2Qyzdxdx,d2Qyzdydy,d2Qyzdxdy,d2Qyzdzdz,d2Qyzdxdz,d2Qyzdydz;
  double DGxx,DGyy,DGzz,DGxy,DGxz,DGyz,TrG,divQx,divQy,divQz;
  double DGchol1xx,DGchol1xy,DGchol1xz,DGchol1yx,DGchol1yy,DGchol1yz;
  double DGchol1zx,DGchol1zy,DGchol1zz,DGchol2xx,DGchol2xy,DGchol2xz;
  double DGchol2yx,DGchol2yy,DGchol2yz,DGchol2zx,DGchol2zy,DGchol2zz;
  double sigxx,sigyy,sigxy,sigxz,sigyz,sigzz;
  double TrQE2,avestress;
  double dEzdx,dEzdy,dEzdz;
  double Qsqxx,Qsqxy,Qsqxz,Qsqyy,Qsqyz,Qsqzz,Qxxl,Qxyl,Qxzl,Qyyl,Qyzl,Qzzl;
  double Hxx,Hyy,Hxy,Hxz,Hyz,TrDQI;
  double duxdx,duxdy,duxdz,duydx,duydy,duydz,duzdx,duzdy,duzdz,Gammap;
  double mDQ4xx,mDQ4xy,mDQ4yy,mDQ4xz,mDQ4yz,mDQ4zz,nnxxl,nnyyl;
  double t1, t2;
  double one_gradient,two_gradient;

#ifdef _BINARY_IO_
  int ibuf[3];     /* For convenvient output of stress information */
  double rbuf[9];
#endif

  int ioff = 0, joff = 0, koff = 0;

#ifdef PARALLEL
  ioff = Lx*pe_cartesian_coordinates_[0]/pe_cartesian_size_[0];
  joff = Ly*pe_cartesian_coordinates_[1]/pe_cartesian_size_[1];
  koff = Lz*pe_cartesian_coordinates_[2]/pe_cartesian_size_[2];
#endif

  for (i=ix1; i<ix2; i++) {
    for (j=jy1; j<jy2; j++) {
      for (k=kz1; k<kz2; k++) {
	density[i][j][k]=0.0;
        freeenergy=0.0;
        freeenergytwist=0.0;
	avestress=0.0;
	for (p=0; p<15; p++) {
	  density[i][j][k] += f[i][j][k][p];
	  u[i][j][k][0] += f[i][j][k][p]*e[p][0];
	  u[i][j][k][1] += f[i][j][k][p]*e[p][1];
	  u[i][j][k][2] += f[i][j][k][p]*e[p][2];
	}
      }
    }
  }

#ifdef PARALLEL
  exchangeMomentumAndQTensor();
#endif  

 one_gradient=0.0;
 two_gradient=0.0;

  for (i=ix1; i<ix2; i++) {
    iup=i+1;
    idwn=i-1;
    for (j=jy1; j<jy2; j++) {
      jup=j+1;
      jdwn=j-1;
      for (k=kz1; k<kz2; k++) {
	kup=k+1;
	kdwn=k-1;

/* first order derivative in the bulk */

	dQxxdx=(Qxx[iup][j][k]-Qxx[idwn][j][k])/2.0;
	dQxydx=(Qxy[iup][j][k]-Qxy[idwn][j][k])/2.0;
	dQxzdx=(Qxz[iup][j][k]-Qxz[idwn][j][k])/2.0;

	dQyxdx=dQxydx;
	dQyydx=(Qyy[iup][j][k]-Qyy[idwn][j][k])/2.0;
	dQyzdx=(Qyz[iup][j][k]-Qyz[idwn][j][k])/2.0;

	dQzxdx=dQxzdx;
	dQzydx=dQyzdx;
	dQzzdx=-(dQxxdx+dQyydx);

	dQxxdy=(Qxx[i][jup][k]-Qxx[i][jdwn][k])/2.0;
	dQxydy=(Qxy[i][jup][k]-Qxy[i][jdwn][k])/2.0;
	dQxzdy=(Qxz[i][jup][k]-Qxz[i][jdwn][k])/2.0;

	dQyxdy=dQxydy;
	dQyydy=(Qyy[i][jup][k]-Qyy[i][jdwn][k])/2.0;
	dQyzdy=(Qyz[i][jup][k]-Qyz[i][jdwn][k])/2.0;

	dQzxdy=dQxzdy;
	dQzydy=dQyzdy;
	dQzzdy=-(dQxxdy+dQyydy);

	dQxxdz=(Qxx[i][j][kup]-Qxx[i][j][kdwn])/2.0;
	dQxydz=(Qxy[i][j][kup]-Qxy[i][j][kdwn])/2.0;
	dQxzdz=(Qxz[i][j][kup]-Qxz[i][j][kdwn])/2.0;

	dQyxdz=dQxydz;
	dQyydz=(Qyy[i][j][kup]-Qyy[i][j][kdwn])/2.0;
	dQyzdz=(Qyz[i][j][kup]-Qyz[i][j][kdwn])/2.0;

	dQzxdz=dQxzdz;
	dQzydz=dQyzdz;
	dQzzdz=-(dQxxdz+dQyydz);

/* second order derivative in the bulk */

	d2Qxxdxdx=Qxx[iup][j][k]-2.0*Qxx[i][j][k]+Qxx[idwn][j][k];
	d2Qxxdydy=Qxx[i][jup][k]-2.0*Qxx[i][j][k]+Qxx[i][jdwn][k];
	d2Qxxdzdz=Qxx[i][j][kup]-2.0*Qxx[i][j][k]+Qxx[i][j][kdwn];
	d2Qxxdxdy=(Qxx[iup][jup][k]-Qxx[iup][jdwn][k]-
		   Qxx[idwn][jup][k]+Qxx[idwn][jdwn][k])/4.0;
	d2Qxxdxdz=(Qxx[iup][j][kup]-Qxx[iup][j][kdwn]-
		   Qxx[idwn][j][kup]+Qxx[idwn][j][kdwn])/4.0;
	d2Qxxdydz=(Qxx[i][jup][kup]-Qxx[i][jup][kdwn]-
		   Qxx[i][jdwn][kup]+Qxx[i][jdwn][kdwn])/4.0;
	

	d2Qyydxdx=Qyy[iup][j][k]-2.0*Qyy[i][j][k]+Qyy[idwn][j][k];
	d2Qyydydy=Qyy[i][jup][k]-2.0*Qyy[i][j][k]+Qyy[i][jdwn][k];
	d2Qyydzdz=Qyy[i][j][kup]-2.0*Qyy[i][j][k]+Qyy[i][j][kdwn];
	d2Qyydxdy=(Qyy[iup][jup][k]-Qyy[iup][jdwn][k]-
		   Qyy[idwn][jup][k]+Qyy[idwn][jdwn][k])/4.0;
	d2Qyydxdz=(Qyy[iup][j][kup]-Qyy[iup][j][kdwn]-
		   Qyy[idwn][j][kup]+Qyy[idwn][j][kdwn])/4.0;
	d2Qyydydz=(Qyy[i][jup][kup]-Qyy[i][jup][kdwn]-
		   Qyy[i][jdwn][kup]+Qyy[i][jdwn][kdwn])/4.0;
	


	d2Qxydxdx=Qxy[iup][j][k]-2.0*Qxy[i][j][k]+Qxy[idwn][j][k];
	d2Qxydydy=Qxy[i][jup][k]-2.0*Qxy[i][j][k]+Qxy[i][jdwn][k];
	d2Qxydzdz=Qxy[i][j][kup]-2.0*Qxy[i][j][k]+Qxy[i][j][kdwn];
	d2Qxydxdy=(Qxy[iup][jup][k]-Qxy[iup][jdwn][k]-
		   Qxy[idwn][jup][k]+Qxy[idwn][jdwn][k])/4.0;
	d2Qxydxdz=(Qxy[iup][j][kup]-Qxy[iup][j][kdwn]-
		   Qxy[idwn][j][kup]+Qxy[idwn][j][kdwn])/4.0;
	d2Qxydydz=(Qxy[i][jup][kup]-Qxy[i][jup][kdwn]-
		   Qxy[i][jdwn][kup]+Qxy[i][jdwn][kdwn])/4.0;
	

	d2Qxzdxdx=Qxz[iup][j][k]-2.0*Qxz[i][j][k]+Qxz[idwn][j][k];
	d2Qxzdydy=Qxz[i][jup][k]-2.0*Qxz[i][j][k]+Qxz[i][jdwn][k];
	d2Qxzdzdz=Qxz[i][j][kup]-2.0*Qxz[i][j][k]+Qxz[i][j][kdwn];
	d2Qxzdxdy=(Qxz[iup][jup][k]-Qxz[iup][jdwn][k]-
		   Qxz[idwn][jup][k]+Qxz[idwn][jdwn][k])/4.0;
	d2Qxzdxdz=(Qxz[iup][j][kup]-Qxz[iup][j][kdwn]-
		   Qxz[idwn][j][kup]+Qxz[idwn][j][kdwn])/4.0;
	d2Qxzdydz=(Qxz[i][jup][kup]-Qxz[i][jup][kdwn]-
		   Qxz[i][jdwn][kup]+Qxz[i][jdwn][kdwn])/4.0;
	

	d2Qyzdxdx=Qyz[iup][j][k]-2.0*Qyz[i][j][k]+Qyz[idwn][j][k];
	d2Qyzdydy=Qyz[i][jup][k]-2.0*Qyz[i][j][k]+Qyz[i][jdwn][k];
	d2Qyzdzdz=Qyz[i][j][kup]-2.0*Qyz[i][j][k]+Qyz[i][j][kdwn];
	d2Qyzdxdy=(Qyz[iup][jup][k]-Qyz[iup][jdwn][k]-
		   Qyz[idwn][jup][k]+Qyz[idwn][jdwn][k])/4.0;
	d2Qyzdxdz=(Qyz[iup][j][kup]-Qyz[iup][j][kdwn]-
		   Qyz[idwn][j][kup]+Qyz[idwn][j][kdwn])/4.0;
	d2Qyzdydz=(Qyz[i][jup][kup]-Qyz[i][jup][kdwn]-
		   Qyz[i][jdwn][kup]+Qyz[i][jdwn][kdwn])/4.0;

/* boundary corrections */
	/*B.C.; use one-sided derivatives*/
	if(pouiseuille1==2){
#if BC
	if(k==0) {
	  dQxxdz= (-3.0*Qxx[i][j][k]+4.0*Qxx[i][j][k+1]-Qxx[i][j][k+2])/2.0;
	  dQxydz= (-3.0*Qxy[i][j][k]+4.0*Qxy[i][j][k+1]-Qxy[i][j][k+2])/2.0; 
	  dQyydz= (-3.0*Qyy[i][j][k]+4.0*Qyy[i][j][k+1]-Qyy[i][j][k+2])/2.0;
	  dQxzdz= (-3.0*Qxz[i][j][k]+4.0*Qxz[i][j][k+1]-Qxz[i][j][k+2])/2.0; 
	  dQyzdz= (-3.0*Qyz[i][j][k]+4.0*Qyz[i][j][k+1]-Qyz[i][j][k+2])/2.0;

	  d2Qxxdzdz= -Qxx[i][j][k+3]+4.0*Qxx[i][j][k+2]-
	    5.0*Qxx[i][j][kup]+2.0*Qxx[i][j][k];
	  d2Qxydzdz= -Qxy[i][j][k+3]+4.0*Qxy[i][j][k+2]-
	    5.0*Qxy[i][j][kup]+2.0*Qxy[i][j][k];
	  d2Qyydzdz= -Qyy[i][j][k+3]+4.0*Qyy[i][j][k+2]-
	    5.0*Qyy[i][j][kup]+2.0*Qyy[i][j][k];
	  d2Qxzdzdz= -Qxz[i][j][k+3]+4.0*Qxz[i][j][k+2]-
	    5.0*Qxz[i][j][kup]+2.0*Qxz[i][j][k];
	  d2Qyzdzdz= -Qyz[i][j][k+3]+4.0*Qyz[i][j][k+2]-
	    5.0*Qyz[i][j][kup]+2.0*Qyz[i][j][k];

	  d2Qxxdxdz=(-3.0*Qxx[iup][j][k]+4.0*Qxx[iup][j][k+1]-Qxx[iup][j][k+2]+
	      3.0*Qxx[idwn][j][k]-4.0*Qxx[idwn][j][k+1]+Qxx[idwn][j][k+2])/4.0;
	  d2Qxydxdz=(-3.0*Qxy[iup][j][k]+4.0*Qxy[iup][j][k+1]-Qxy[iup][j][k+2]+
	      3.0*Qxy[idwn][j][k]-4.0*Qxy[idwn][j][k+1]+Qxy[idwn][j][k+2])/4.0;
	  d2Qyydxdz=(-3.0*Qyy[iup][j][k]+4.0*Qyy[iup][j][k+1]-Qyy[iup][j][k+2]+
	      3.0*Qyy[idwn][j][k]-4.0*Qyy[idwn][j][k+1]+Qyy[idwn][j][k+2])/4.0;
	  d2Qxzdxdz=(-3.0*Qxz[iup][j][k]+4.0*Qxz[iup][j][k+1]-Qxz[iup][j][k+2]+
	      3.0*Qxz[idwn][j][k]-4.0*Qxz[idwn][j][k+1]+Qxz[idwn][j][k+2])/4.0;
	  d2Qyzdxdz=(-3.0*Qyz[iup][j][k]+4.0*Qyz[iup][j][k+1]-Qyz[iup][j][k+2]+
	      3.0*Qyz[idwn][j][k]-4.0*Qyz[idwn][j][k+1]+Qyz[idwn][j][k+2])/4.0;

	  d2Qxxdydz=(-3.0*Qxx[i][jup][k]+4.0*Qxx[i][jup][k+1]-Qxx[i][jup][k+2]+
	      3.0*Qxx[i][jdwn][k]-4.0*Qxx[i][jdwn][k+1]+Qxx[i][jdwn][k+2])/4.0;
	  d2Qxydydz=(-3.0*Qxy[i][jup][k]+4.0*Qxy[i][jup][k+1]-Qxy[i][jup][k+2]+
	      3.0*Qxy[i][jdwn][k]-4.0*Qxy[i][jdwn][k+1]+Qxy[i][jdwn][k+2])/4.0;
	  d2Qyydydz=(-3.0*Qyy[i][jup][k]+4.0*Qyy[i][jup][k+1]-Qyy[i][jup][k+2]+
	      3.0*Qyy[i][jdwn][k]-4.0*Qyy[i][jdwn][k+1]+Qyy[i][jdwn][k+2])/4.0;
	  d2Qxzdydz=(-3.0*Qxz[i][jup][k]+4.0*Qxz[i][jup][k+1]-Qxz[i][jup][k+2]+
	      3.0*Qxz[i][jdwn][k]-4.0*Qxz[i][jdwn][k+1]+Qxz[i][jdwn][k+2])/4.0;
	  d2Qyzdydz=(-3.0*Qyz[i][jup][k]+4.0*Qyz[i][jup][k+1]-Qyz[i][jup][k+2]+
	      3.0*Qyz[i][jdwn][k]-4.0*Qyz[i][jdwn][k+1]+Qyz[i][jdwn][k+2])/4.0;

	}
	else if(k==Lz-1) {
	  dQxxdz=(3.0*Qxx[i][j][k]-4.0*Qxx[i][j][k-1]+Qxx[i][j][k-2])/2.0;
	  dQxydz=(3.0*Qxy[i][j][k]-4.0*Qxy[i][j][k-1]+Qxy[i][j][k-2])/2.0; 
	  dQyydz=(3.0*Qyy[i][j][k]-4.0*Qyy[i][j][k-1]+Qyy[i][j][k-2])/2.0;
	  dQxzdz=(3.0*Qxz[i][j][k]-4.0*Qxz[i][j][k-1]+Qxz[i][j][k-2])/2.0; 
	  dQyzdz=(3.0*Qyz[i][j][k]-4.0*Qyz[i][j][k-1]+Qyz[i][j][k-2])/2.0;

	  d2Qxxdzdz= -Qxx[i][j][k-3]+4.0*Qxx[i][j][k-2]-
	    5.0*Qxx[i][j][kdwn]+2.0*Qxx[i][j][k];
	  d2Qxydzdz= -Qxy[i][j][k-3]+4.0*Qxy[i][j][k-2]-
	    5.0*Qxy[i][j][kdwn]+2.0*Qxy[i][j][k];
	  d2Qyydzdz= -Qyy[i][j][k-3]+4.0*Qyy[i][j][k-2]-
	    5.0*Qyy[i][j][kdwn]+2.0*Qyy[i][j][k];
	  d2Qxzdzdz= -Qxz[i][j][k-3]+4.0*Qxz[i][j][k-2]-
	    5.0*Qxz[i][j][kdwn]+2.0*Qxz[i][j][k];
	  d2Qyzdzdz= -Qyz[i][j][k-3]+4.0*Qyz[i][j][k-2]-
	    5.0*Qyz[i][j][kdwn]+2.0*Qyz[i][j][k];

	  d2Qxxdxdz=(3.0*Qxx[iup][j][k]-4.0*Qxx[iup][j][k-1]+Qxx[iup][j][k-2]-
	      3.0*Qxx[idwn][j][k]+4.0*Qxx[idwn][j][k-1]-Qxx[idwn][j][k-2])/4.0;
	  d2Qxydxdz=(3.0*Qxy[iup][j][k]-4.0*Qxy[iup][j][k-1]+Qxy[iup][j][k-2]-
	      3.0*Qxy[idwn][j][k]+4.0*Qxy[idwn][j][k-1]-Qxy[idwn][j][k-2])/4.0;
	  d2Qyydxdz=(3.0*Qyy[iup][j][k]-4.0*Qyy[iup][j][k-1]+Qyy[iup][j][k-2]-
	      3.0*Qyy[idwn][j][k]+4.0*Qyy[idwn][j][k-1]-Qyy[idwn][j][k-2])/4.0;
	  d2Qxzdxdz=(3.0*Qxz[iup][j][k]-4.0*Qxz[iup][j][k-1]+Qxz[iup][j][k-2]-
	      3.0*Qxz[idwn][j][k]+4.0*Qxz[idwn][j][k-1]-Qxz[idwn][j][k-2])/4.0;
	  d2Qyzdxdz=(3.0*Qyz[iup][j][k]-4.0*Qyz[iup][j][k-1]+Qyz[iup][j][k-2]-
	      3.0*Qyz[idwn][j][k]+4.0*Qyz[idwn][j][k-1]-Qyz[idwn][j][k-2])/4.0;

	  d2Qxxdydz=(3.0*Qxx[i][jup][k]-4.0*Qxx[i][jup][k-1]+Qxx[i][jup][k-2]-
	      3.0*Qxx[i][jdwn][k]+4.0*Qxx[i][jdwn][k-1]-Qxx[i][jdwn][k-2])/4.0;
	  d2Qxydydz=(3.0*Qxy[i][jup][k]-4.0*Qxy[i][jup][k-1]+Qxy[i][jup][k-2]-
	      3.0*Qxy[i][jdwn][k]+4.0*Qxy[i][jdwn][k-1]-Qxy[i][jdwn][k-2])/4.0;
	  d2Qyydydz=(3.0*Qyy[i][jup][k]-4.0*Qyy[i][jup][k-1]+Qyy[i][jup][k-2]-
	      3.0*Qyy[i][jdwn][k]+4.0*Qyy[i][jdwn][k-1]-Qyy[i][jdwn][k-2])/4.0;
	  d2Qxzdydz=(3.0*Qxz[i][jup][k]-4.0*Qxz[i][jup][k-1]+Qxz[i][jup][k-2]-
	      3.0*Qxz[i][jdwn][k]+4.0*Qxz[i][jdwn][k-1]-Qxz[i][jdwn][k-2])/4.0;
	  d2Qyzdydz=(3.0*Qyz[i][jup][k]-4.0*Qyz[i][jup][k-1]+Qyz[i][jup][k-2]-
	      3.0*Qyz[i][jdwn][k]+4.0*Qyz[i][jdwn][k-1]-Qyz[i][jdwn][k-2])/4.0;

	}
#endif
	}

	duydz=(u[i][j][kup][1]-u[i][j][kdwn][1])/2.0;
	
      /*B.C.; use one-sided derivatives*/
	if(pouiseuille1==1){
#if BC
	if(k==0) {
	  duydz= 0.0*(-3.0*u[i][j][k][1]+4.0*u[i][j][k+1][1]-u[i][j][k+2][1])/2.0;
	}
	else if(k==Lz-1) {
	  duydz= 0.0*(3.0*u[i][j][k][1]-4.0*u[i][j][k-1][1]+u[i][j][k-2][1])/2.0;
	  
	}
#endif
	}

/* \parial F / \partial dQ * dQ */

	DGxx=(dQxxdx*dQxxdx+2.0*dQxydx*dQxydx+dQyydx*dQyydx+
	      2.0*dQxzdx*dQxzdx+2.0*dQyzdx*dQyzdx+
	      (dQxxdx+dQyydx)*(dQxxdx+dQyydx));
	DGyy= (dQxxdy*dQxxdy+2.0*dQxydy*dQxydy+dQyydy*dQyydy+
	       2.0*dQxzdy*dQxzdy+2.0*dQyzdy*dQyzdy+
	       (dQxxdy+dQyydy)*(dQxxdy+dQyydy));
	DGzz= (dQxxdz*dQxxdz+2.0*dQxydz*dQxydz+dQyydz*dQyydz+
	       2.0*dQxzdz*dQxzdz+2.0*dQyzdz*dQyzdz+
	       (dQxxdz+dQyydz)*(dQxxdz+dQyydz));
	TrG = (DGxx+DGyy+DGzz)/3.0;
	DGxy= (dQxxdx*dQxxdy+2.0*dQxydx*dQxydy+dQyydx*dQyydy+
	       2.0*dQxzdx*dQxzdy+2.0*dQyzdx*dQyzdy+
	       (dQxxdx+dQyydx)*(dQxxdy+dQyydy));
	DGxz= (dQxxdx*dQxxdz+2.0*dQxydx*dQxydz+dQyydx*dQyydz+
	       2.0*dQxzdx*dQxzdz+2.0*dQyzdx*dQyzdz+
	       (dQxxdx+dQyydx)*(dQxxdz+dQyydz));
	DGyz= (dQxxdy*dQxxdz+2.0*dQxydy*dQxydz+dQyydy*dQyydz+
	       2.0*dQxzdy*dQxzdz+2.0*dQyzdy*dQyzdz+
	       (dQxxdy+dQyydy)*(dQxxdz+dQyydz));

        DGchol1xx=1.0/2.0*
        (2.0*dQxxdx*dQxxdx+2.0*(dQxydx*dQxydx+dQxxdy*dQxydx)+
         2.0*(dQxzdx*dQxzdx+dQxxdz*dQxzdx)+2.0*(dQxydy*dQyydx)+
         2.0*(dQxzdy*dQyzdx+dQxydz*dQyzdx)+2.0*(-dQxzdz*dQxxdx
         -dQxzdz*dQyydx));
        DGchol1xy=1.0/2.0*
        (2.0*(dQxxdx*dQxxdy)+2.0*(dQxydx*dQxydy+dQxxdy*dQxydy)
        +2.0*(dQxzdx*dQxzdy+dQxxdz*dQxzdy)+2.0*(dQxydy*dQyydy)
        +2.0*(dQxzdy*dQyzdy+dQxydz*dQyzdy)+2.0*(-dQxzdz*dQxxdy
         -dQxzdz*dQyydy));
        DGchol1xz=1.0/2.0*
        (2.0*(dQxxdx*dQxxdz)+2.0*(dQxydx*dQxydz+dQxxdy*dQxydz)
        +2.0*(dQxzdx*dQxzdz+dQxxdz*dQxzdz)+2.0*(dQxydy*dQyydz)
        +2.0*(dQxzdy+dQxydz)*dQyzdz+2.0*(-dQxxdz-dQyydz)*dQxzdz);        
        DGchol1yx=1.0/2.0*
        (2.0*(dQxydx*dQxxdx)+2.0*(dQyydx+dQxydy)*dQxydx
        +2.0*(dQyzdx+dQxydz)*dQxzdx+2.0*(dQyydy*dQyydx)
        +2.0*(dQyzdy+dQyydz)*dQyzdx+2.0*(-dQxxdx-dQyydx)*dQyzdz);        
        DGchol1yy=1.0/2.0*
        (2.0*(dQxydx*dQxxdy)+2.0*(dQyydx+dQxydy)*dQxydy
        +2.0*(dQyzdx+dQxydz)*dQxzdy+2.0*dQyydy*dQyydy
        +2.0*(dQyzdy+dQyydz)*dQyzdy+2.0*dQyzdz*(-dQxxdy-dQyydy));
        DGchol1yz=1.0/2.0*
        (2.0*dQxydx*dQxxdz+2.0*(dQxydy+dQyydx)*dQxydz
        +2.0*(dQyzdx+dQxydz)*dQxzdz+2.0*dQyydy*dQyydz
        +2.0*(dQyzdy+dQyydz)*dQyzdz+2.0*dQyzdz*(-dQxxdz-dQyydz));
        DGchol1zx=1.0/2.0*
        (2.0*dQxzdx*dQxxdx+2.0*(dQyzdx+dQxzdy)*dQxydx
        +2.0*(-dQxxdx-dQyydx+dQxzdz)*dQxzdx+2.0*dQyzdy*dQyydx
        +2.0*(-dQxxdy-dQyydy+dQyzdz)*dQyzdx+2.0*(-dQxxdz-dQyydz)*
        (-dQxxdx-dQyydx));
        DGchol1zy=1.0/2.0*
        (2.0*dQxzdx*dQxxdy+2.0*(dQyzdx+dQxzdy)*dQxydy
        +2.0*(-dQxxdx-dQyydx+dQxzdz)*dQxzdy+2.0*(dQyzdy*dQyydy)
        +2.0*(-dQxxdy-dQyydy+dQyzdz)*dQyzdy+2.0*(-dQxxdz-dQyydz)*
        (-dQxxdy-dQyydy));        
        DGchol1zz=1.0/2.0*
        (2.0*dQxzdx*dQxxdz+2.0*(dQyzdx+dQxzdy)*dQxydz
        +2.0*(-dQxxdx-dQyydx+dQxzdz)*dQxzdz+2.0*(dQyzdy*dQyydz)
        +2.0*(-dQxxdy-dQyydy+dQyzdz)*dQyzdz+2.0*(-dQxxdz-dQyydz)*
        (-dQxxdz-dQyydz));
        DGchol2xx= (2.0*Qxz[i][j][k]*dQxydx+(2.0*Qyz[i][j][k]*dQyydx)
        +2.0*(-Qxx[i][j][k]-2.0*Qyy[i][j][k])*dQyzdx
        +2.0*(-Qxy[i][j][k])*dQxzdx+2.0*(-Qyz[i][j][k])*
        (-dQxxdx-dQyydx)) ;
        DGchol2xy= (2.0*Qxz[i][j][k]*dQxydy+(2.0*Qyz[i][j][k]*dQyydy)
        +2.0*(-Qxx[i][j][k]-2.0*Qyy[i][j][k])*dQyzdy
        +2.0*(-Qxy[i][j][k])*dQxzdy+2.0*(-Qyz[i][j][k])*(-dQxxdy-dQyydy));
        DGchol2xz= (2.0*Qxz[i][j][k]*dQxydz+(2.0*Qyz[i][j][k]*dQyydz)
        +2.0*(-Qxx[i][j][k]-2.0*Qyy[i][j][k])*dQyzdz
        +2.0*(-Qxy[i][j][k])*dQxzdz+2.0*(-Qyz[i][j][k])*(-dQxxdz-dQyydz));
        DGchol2yx= (2.0*(-Qxz[i][j][k])*dQxxdx+2.0*(-Qyz[i][j][k])*dQxydx
        +2.0*(2.0*Qxx[i][j][k]+Qyy[i][j][k])*dQxzdx+2.0*(Qxy[i][j][k])*dQyzdx
        +2.0*Qxz[i][j][k]*(-dQxxdx-dQyydx));
        DGchol2yy= (2.0*(-Qxz[i][j][k])*dQxxdy+2.0*(-Qyz[i][j][k])*dQxydy
        +2.0*(2.0*Qxx[i][j][k]+Qyy[i][j][k])*dQxzdy+2.0*(Qxy[i][j][k])*dQyzdy
        +2.0*Qxz[i][j][k]*(-dQxxdy-dQyydy));
        DGchol2yz= (2.0*(-Qxz[i][j][k])*dQxxdz+2.0*(-Qyz[i][j][k])*dQxydz+
           2.0*(2.0*Qxx[i][j][k]+Qyy[i][j][k])*dQxzdz+2.0*(Qxy[i][j][k])
           *dQyzdz+
           2.0*Qxz[i][j][k]*(-dQxxdz-dQyydz));
        DGchol2zx= (2.0*(Qxy[i][j][k])*dQxxdx+2.0*(Qyy[i][j][k]
           -Qxx[i][j][k])*dQxydx+
           2.0*(Qyz[i][j][k])*dQxzdx-2.0*(Qxy[i][j][k])*dQyydx
           +2.0*(-Qxz[i][j][k])*dQyzdx);
        DGchol2zy= (2.0*(Qxy[i][j][k])*dQxxdy+2.0*(Qyy[i][j][k]
           -Qxx[i][j][k])*dQxydy+
           2.0*(Qyz[i][j][k])*dQxzdy-2.0*Qxy[i][j][k]
           *dQyydy+2.0*(-Qxz[i][j][k])*dQyzdy);
        DGchol2zz= (2.0*(Qxy[i][j][k])*dQxxdz+2.0*(Qyy[i][j][k]
           -Qxx[i][j][k])*dQxydz+
           2.0*(Qyz[i][j][k])*dQxzdz
           -2.0*Qxy[i][j][k]*dQyydz+2.0*(-Qxz[i][j][k])*dQyzdz);

	divQx=dQxxdx+dQxydy+dQxzdz;
	divQy=dQxydx+dQyydy+dQyzdz;
	divQz=dQxzdx+dQyzdy-dQxxdz-dQyydz;

/* summing several terms up for the stress */

       DG2xx[i][j][k]= L1/2.0*DGxx-L1/2.0*DGchol1xx+
          L2*(dQxxdx*divQx+dQxydx*divQy+dQxzdx*divQz)
          +L1/2.0*DGchol1yy+L1/2.0*DGchol1zz
          -L1/2.0*DGyy-L1/2.0*DGzz
          -L1*q0*DGchol2yy-L1*q0*DGchol2zz
          -L2/2.0*(divQx*divQx+divQy*divQy+divQz*divQz);
        DG2yy[i][j][k]= L1/2.0*DGyy-L1/2.0*DGchol1yy+
          L2*(dQxydy*divQx+dQyydy*divQy+dQyzdy*divQz)
          +L1/2.0*DGchol1xx+L1/2.0*DGchol1zz
          -L1/2.0*DGxx-L1/2.0*DGzz
          -L1*q0*DGchol2xx-L1*q0*DGchol2zz
          -L2/2.0*(divQx*divQx+divQy*divQy+divQz*divQz);
        DG2zz[i][j][k]= L1/2.0*DGzz-L1/2.0*DGchol1zz+
          L2*(dQxzdz*divQx+dQyzdz*divQy-(dQxxdz+dQyydz)*divQz)
          +L1/2.0*DGchol1xx+L1/2.0*DGchol1yy
          -L1/2.0*DGxx-L1/2.0*DGyy
          -L1*q0*DGchol2xx-L1*q0*DGchol2yy
          -L2/2.0*(divQx*divQx+divQy*divQy+divQz*divQz);
        DG2xy[i][j][k]= L1*DGxy-L1*DGchol1xy+L1*q0*DGchol2xy+
          L2*(dQxxdy*divQx+dQxydy*divQy+dQxzdy*divQz);
        DG2xz[i][j][k]= L1*DGxz-L1*DGchol1xz+L1*q0*DGchol2xz+
          L2*(dQxxdz*divQx+dQxydz*divQy+dQxzdz*divQz);
        DG2yz[i][j][k]= L1*DGyz-L1*DGchol1yz+L1*q0*DGchol2yz+
          L2*(dQxydz*divQx+dQyydz*divQy+dQyzdz*divQz);
	DG2yx[i][j][k] = L1*DGxy-L1*DGchol1yx+L1*q0*DGchol2yx+
	  L2*(dQxydx*divQx+dQyydx*divQy+dQyzdx*divQz);
	DG2zx[i][j][k] = L1*DGxz-L1*DGchol1zx+L1*q0*DGchol2zx+
	  L2*(dQxzdx*divQx+dQyzdx*divQy-(dQxxdx+dQyydx)*divQz);
	DG2zy[i][j][k] = L1*DGyz-L1*DGchol1zy+L1*q0*DGchol2zy+
	  L2*(dQxzdy*divQx+dQyzdy*divQy-(dQxxdy+dQyydy)*divQz);

/* order parameter gradient contribution to the stress */

	Stressxx[i][j][k]=-DG2xx[i][j][k];
	Stressxy[i][j][k]=-DG2yx[i][j][k];
	Stressxz[i][j][k]=-DG2zx[i][j][k];
	Stressyy[i][j][k]=-DG2yy[i][j][k];
	Stressyz[i][j][k]=-DG2zy[i][j][k];
	Stresszz[i][j][k]=-DG2zz[i][j][k];
	Stressyx[i][j][k]=-DG2xy[i][j][k];
	Stresszx[i][j][k]=-DG2xz[i][j][k];
	Stresszy[i][j][k]=-DG2yz[i][j][k];


      Qxxl=Qxx[i][j][k];
      Qxyl=Qxy[i][j][k];
      Qyyl=Qyy[i][j][k];
      Qxzl=Qxz[i][j][k];
      Qyzl=Qyz[i][j][k];
      Qzzl=-Qxxl-Qyyl;

      nnxxl= Qxxl+1.0/3.0;
      nnyyl= Qyyl+1.0/3.0;


      Qsqxx=Qxxl+Qyyl;
      Qsqzz=Qsqxx*Qsqxx+Qxzl*Qxzl+Qyzl*Qyzl;
      Qsqxy=Qxyl*Qsqxx+Qxzl*Qyzl;
      Qsqxz=Qxxl*Qxzl-Qxzl*Qsqxx+Qxyl*Qyzl;
      Qsqyz=Qxyl*Qxzl-Qyzl*Qsqxx+Qyyl*Qyzl;
      Qsqxx=Qxxl*Qxxl+Qxyl*Qxyl+Qxzl*Qxzl;
      Qsqyy=Qyyl*Qyyl+Qxyl*Qxyl+Qyzl*Qyzl;
             
 
      Hxx=molfieldxx[i][j][k];
      Hxy=molfieldxy[i][j][k];
      Hxz=molfieldxz[i][j][k];
      Hyy=molfieldyy[i][j][k];
      Hyz=molfieldyz[i][j][k];


#if BE  
/* molecular field contribution to the stress */
   
	sigxx=2.0/3.0*xi*((1.0+3.0*Qxxl)*
		  (Hxx*(1.0-2.0*Qxxl-Qyyl)-Hyy*(Qxxl+2.0*Qyyl)-2.0*Hyz*Qyzl)+
		  (Hxy*Qxyl+Hxz*Qxzl)*(1.0-6.0*Qxxl));
	sigxy=xi*(Hxy*(2.0/3.0+Qxxl-4.0*Qxyl*Qxyl+Qyyl)-
		  Hxx*Qxyl*(-1.0+4.0*Qxxl+2.0*Qyyl)-
		  Hyy*Qxyl*(-1.0+4.0*Qyyl+2.0*Qxxl)+
		  Hxz*(-4.0*Qxyl*Qxzl+Qyzl)+Hyz*(Qxzl-4.0*Qxyl*Qyzl));
	sigyy=2.0/3.0*xi*((1.0+3.0*Qyyl)*
		  (Hyy*(1.0-Qxxl-2.0*Qyyl)-Hxx*(2.0*Qxxl+Qyyl)-2.0*Hxz*Qxzl)+
		  (Hxy*Qxyl+Hyz*Qyzl)*(1.0-6.0*Qyyl));
	sigxz=xi*(Hxz*(2.0/3.0-4.0*Qxzl*Qxzl-Qyyl)-
		  Hxx*Qxzl*(4.0*Qxxl+2.0*Qyyl)-
		  Hyy*Qxzl*(1.0+4.0*Qyyl+2.0*Qxxl)+
		  Hxy*(Qyzl-4.0*Qxyl*Qxzl)+Hyz*(Qxyl-4.0*Qxzl*Qyzl));
	sigyz=xi*(Hyz*(2.0/3.0-4.0*Qyzl*Qyzl-Qxxl)-
		  Hyy*Qyzl*(4.0*Qyyl+2.0*Qxxl)-
		  Hxx*Qyzl*(1.0+4.0*Qxxl+2.0*Qyyl)+
		  Hxy*(Qxzl-4.0*Qxyl*Qyzl)+Hxz*(Qxyl-4.0*Qxzl*Qyzl));
#else
	sigxx=2.0/3.0*xi*Hxx;
	sigxy=2.0/3.0*xi*Hxy;
	sigyy=2.0/3.0*xi*Hyy;
	sigxz=2.0/3.0*xi*Hxz;
	sigyz=2.0/3.0*xi*Hyz;
#endif


	Stressxx[i][j][k]+=-sigxx;
	Stressxy[i][j][k]+=-sigxy;
	Stressxz[i][j][k]+=-sigxz;
	Stressyy[i][j][k]+=-sigyy;
	Stressyz[i][j][k]+=-sigyz;
	Stresszz[i][j][k]+=sigxx+sigyy;
	Stressyx[i][j][k]+=-sigxy;
	Stresszx[i][j][k]+=-sigxz;
	Stresszy[i][j][k]+=-sigyz;

/* antisymmetric contribution from parametercalc */

	Stressxy[i][j][k]+=tauxy[i][j][k];
	Stressxz[i][j][k]+=tauxz[i][j][k];
	Stressyz[i][j][k]+=tauyz[i][j][k];
	Stressyx[i][j][k]-=tauxy[i][j][k];
	Stresszx[i][j][k]-=tauxz[i][j][k];
	Stresszy[i][j][k]-=tauyz[i][j][k];


// NOTE: the cubic redshift is calculted with L1init, L2init and q0init in contrary to the FE functional. The redshift minimization has the meaning of a partial derivative with respect to the cell size

      if (REDSHIFT==1){
	   two_gradient+=L1init/2.0*(-DGchol1xx+DGxx-DGchol1yy+DGyy-DGchol1zz+DGzz)
	     +L2init/2.0*(divQx*divQx+divQy*divQy+divQz*divQz);
	   one_gradient+=L1init*q0init*(DGchol2xx+DGchol2yy+DGchol2zz);
      }

        freeenergy+=L1/2.0*(-DGchol1xx+DGxx-DGchol1yy+DGyy-DGchol1zz+DGzz)
        +L2/2.0*(divQx*divQx+divQy*divQy+divQz*divQz)
        +L1*q0*(DGchol2xx+DGchol2yy+DGchol2zz);
        freeenergy+=(Abulk/2.0*(1.0-gam/3.0)+2.0*L1*q0*q0)
        *(Qsqxx+Qsqyy+Qsqzz);
        freeenergy+=Abulk*gam/4.0*(Qsqxx+Qsqyy+Qsqzz)*(Qsqxx+
        +Qsqyy+Qsqzz);
        freeenergy+=-Abulk*gam/3.0*(Qsqxx*Qxxl+2.0*Qsqxy*Qxyl
        +2.0*Qsqxz*Qxzl+Qsqyy*Qyyl+2.0*Qsqyz*Qyzl+Qsqzz*Qzzl );

        freeenergytwist+= L1/2.0*(-DGchol1xx+DGxx-DGchol1yy+
        DGyy-DGchol1zz+DGzz)+L1*q0*(DGchol2xx+DGchol2yy+DGchol2zz)
        +2.0*q0*q0*L1*(Qxxl*Qxxl+2.0*Qxyl*Qxyl+
         2.0*Qxzl*Qxzl+Qyyl*Qyyl+2.0*Qyzl*Qyzl+Qzzl*Qzzl);


	 // electric field contribution
#ifdef EON

	 freeenergy+= -Inv12Pi*epsa*(Qxxl*Ex[i][j][k]*Ex[i][j][k]+2.0*Qxyl*Ex[i][j][k]*Ey[i][j][k]+ 
	    2.0*Qxzl*Ex[i][j][k]*Ez[i][j][k]+Qyyl*Ey[i][j][k]*Ey[i][j][k]+
	    2.0*Qyzl*Ey[i][j][k]*Ez[i][j][k]+Qzzl*Ez[i][j][k]*Ez[i][j][k]);

#endif


	 }
      }
   }


/* free energy output */

#ifdef PARALLEL

  t1 = MPI_Wtime();

  /* KS. Could replace with single MPI_Reduce(). Not particularly important.*/

    double reducedF,reducedFtwist,reducedstress,reduced_one_gradient,reduced_two_gradient;

    MPI_Allreduce(&freeenergy,&reducedF, 1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    freeenergy=reducedF;
    MPI_Allreduce(&freeenergytwist,&reducedFtwist, 1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    freeenergytwist=reducedFtwist;

    MPI_Allreduce(&one_gradient,&reduced_one_gradient, 1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    one_gradient=reduced_one_gradient;
    MPI_Allreduce(&two_gradient,&reduced_two_gradient, 1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    two_gradient=reduced_two_gradient;

    /* KS not required? */
    MPI_Allreduce(&avestress,&reducedstress, 1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    avestress=reducedstress;
#endif


// redshift: threshold set to 1e-9 as too small redshift cause program crash
   if (REDSHIFT==1){

      rr_old=rr;
      rr=-0.5*one_gradient/two_gradient;

      if(fabs(rr)<1e-9){rr=rr_old;}

   }

    /* Make sure this output comes from the process consistent with
     * the above MPI_Reduce */

/* free energy and redshift output */

   if (n % (FePrintInt) == 0){
       if (myPE == 0) {
	 String fileName("fe.");
	 fileName.concat((int) numCase);
	 fileName.concat(".dat");
	 ofstream file(fileName.get(),ios::app);
	 file.precision(6);
	 file.setf(ios::scientific);
	 file << n << "\t" << freeenergy/Lx/Ly/Lz << " " << freeenergytwist/Lx/Ly/Lz << " " << rr <<"\t" << phivr << endl;
	 file.close();
       }
   }

/* stress output */

    if (n % (FePrintInt*SigPrintFac) == 0){

      String signame("sig.");
      signame.concat(numCase);
      signame.concat(".");
      signame.concat(n);
      signame.concat(".dat-");
      signame.concat(io_group_id_);
      signame.concat("-");
      signame.concat(io_ngroups_);


#ifdef PARALLEL

      int token=0;
      if (io_rank_ == 0) {
	output.open(signame.get());
      }
      else {
	MPI_Recv(&token, 1, MPI_INT, io_rank_ - 1, 0, io_communicator_,
		 &status);
	output.open(signame.get(),ios::app);
      }
#else
      output.open(signame.get());
#endif

      output.precision(5);

      for (i=ix1; i<ix2; i++) {
	for (j=jy1; j<jy2; j++) {
	  for (k=kz1; k<kz2; k++) {

#ifdef _BINARY_IO_
	    ibuf[0] = i-ix1+ioff;
	    ibuf[1] = j-jy1+joff;
	    ibuf[2] = k-kz1+koff;
	    rbuf[0] = Stressxx[i][j][k];
	    rbuf[1] = Stressxy[i][j][k];
	    rbuf[2] = Stressxz[i][j][k];
	    rbuf[3] = Stressyx[i][j][k];
	    rbuf[4] = Stressyy[i][j][k];
	    rbuf[5] = Stressyz[i][j][k];
	    rbuf[6] = Stresszx[i][j][k];
	    rbuf[7] = Stresszy[i][j][k];
	    rbuf[8] = Stresszz[i][j][k];
	    output.write((char *) ibuf, 3*sizeof(int));
	    output.write((char *) rbuf, 9*sizeof(double));

#else
	    /* Use usual ASCII */
	    output << i-ix1+ioff << " " << j-jy1+joff << " " << k-kz1+koff
		   << " " << Stressxx[i][j][k] << " " << Stressxy[i][j][k]
		   << " " << Stressxz[i][j][k]<<  " " << Stressyx[i][j][k]
		   << " " << Stressyy[i][j][k] << " " << Stressyz[i][j][k]
		   << " " << Stresszx[i][j][k] << " " << Stresszy[i][j][k]
		   << " " << Stresszz[i][j][k] << endl;
#endif

      } 
    }
  }

      output.close();


#ifdef PARALLEL
      if (io_rank_ < io_group_size_ - 1) {
	MPI_Send(&token, 1, MPI_INT, io_rank_ + 1, 0, io_communicator_);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      t2 = MPI_Wtime();
      total_io_ += (t2-t1);
#endif

    }

}

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
  double t0, t1;
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
  double t0, t1;
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
 *  The argument is the time step.
 *
 *  This is always binary.
 *
 *****************************************************************************/

void writeRestart(const int iter) {

  int i, j, k, p;  
  double t0, t1;
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

  t0 = MPI_Wtime();

  if (io_rank_ == 0) {
    output.open(oname.get());
  }
  else {
    MPI_Recv(&token, 1, MPI_INT, io_rank_ - 1, tag, io_communicator_,
	     &status);
    output.open(oname.get(),ios::app);
  }
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
 *  the Q tensor) from file.
 *
 *  The argument is the time step.
 *
 *  Binary is always expected.
 *
 *****************************************************************************/

void readRestart(const int iter) {

  int i, j, k, p;
  double t0, t1;
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

#ifdef PARALLEL
  int token=0;
  const int tag = 990;

  t0 = MPI_Wtime();

  if (io_rank_ == 0) {
    input.open(iname.get());
  }
  else {
    /* Block until we get the token */
    MPI_Recv(&token, 1, MPI_INT, io_rank_ - 1, tag, io_communicator_,
	     &status);
    input.open(iname.get(),ios::in);
  }
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

  if (input.good()) {
  }
  else {
    cout << "Error on read" << endl;
    MPI_Abort(MPI_COMM_WORLD, 0);
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
