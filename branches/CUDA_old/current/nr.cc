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


#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);\
	a[k][l]=h+s*(g-h*tau);
#define n 3
void jacobi(double (*a)[n], double d[], double (*v)[n], int *nrot)
{
  int j,iq,ip,i;
  double tresh,theta,tau,t,sm,s,h,g,c;
  double b[n],z[n];

  for (ip=0;ip<n;ip++) {
    for (iq=0;iq<n;iq++) v[ip][iq]=0.0;
    v[ip][ip]=1.0;
  }
  for (ip=0;ip<n;ip++) {
    b[ip]=d[ip]=a[ip][ip];
    z[ip]=0.0;
  }
  *nrot=0;
  for (i=1;i<=50;i++) {
    sm=0.0;
    for (ip=0;ip< n-1;ip++) {
      for (iq=ip+1;iq<n;iq++)
	sm += fabs(a[ip][iq]);
    }
    if (sm == 0.0) {
      return;
    }
    if (i < 4)
      tresh=0.2*sm/(n*n);
    else
      tresh=0.0;
    for (ip=0;ip<n-1;ip++) {
      for (iq=ip+1;iq<n;iq++) {
	g=100.0*fabs(a[ip][iq]);
	if (i > 4 && (fabs(d[ip])+g) == fabs(d[ip])
	    && (fabs(d[iq])+g) == fabs(d[iq]))
	  a[ip][iq]=0.0;
	else if (fabs(a[ip][iq]) > tresh) {
	  h=d[iq]-d[ip];
	  if ((fabs(h)+g) == fabs(h))
	    t=(a[ip][iq])/h;
	  else {
	    theta=0.5*h/(a[ip][iq]);
	    t=1.0/(fabs(theta)+sqrt(1.0+theta*theta));
	    if (theta < 0.0) t = -t;
	  }
	  c=1.0/sqrt(1+t*t);
	  s=t*c;
	  tau=s/(1.0+c);
	  h=t*a[ip][iq];
	  z[ip] -= h;
	  z[iq] += h;
	  d[ip] -= h;
	  d[iq] += h;
	  a[ip][iq]=0.0;
	  for (j=0;j<=ip-1;j++) {
	    ROTATE(a,j,ip,j,iq)
	      }
	  for (j=ip+1;j<=iq-1;j++) {
	    ROTATE(a,ip,j,j,iq)
	      }
	  for (j=iq+1;j<n;j++) {
	    ROTATE(a,ip,j,iq,j)
	      }
	  for (j=0;j<n;j++) {
	    ROTATE(v,j,ip,j,iq)
	      }
	  ++(*nrot);
	}
      }
    }
    for (ip=0;ip<n;ip++) {
      b[ip] += z[ip];
      d[ip]=b[ip];
      z[ip]=0.0;
    }
  }
  cout << "Too many iterations in routine jacobi" << endl;
  exit(0);
}
#undef n
#undef ROTATE


#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
void fourn(double data[], unsigned long nn[], int ndim, int isign)
{
  int idim;
  unsigned long i1,i2,i3,i2rev,i3rev,ip1,ip2,ip3,ifp1,ifp2;
  unsigned long ibit,k1,k2,n,nprev,nrem,ntot;
  double tempi,tempr;
  double theta,wi,wpi,wpr,wr,wtemp;

  for (ntot=1,idim=1;idim<=ndim;idim++)
    ntot *= nn[idim];
  nprev=1;
  for (idim=ndim;idim>=1;idim--) {
    n=nn[idim];
    nrem=ntot/(n*nprev);
    ip1=nprev << 1;
    ip2=ip1*n;
    ip3=ip2*nrem;
    i2rev=1;
    for (i2=1;i2<=ip2;i2+=ip1) {
      if (i2 < i2rev) {
	for (i1=i2;i1<=i2+ip1-2;i1+=2) {
	  for (i3=i1;i3<=ip3;i3+=ip2) {
	    i3rev=i2rev+i3-i2;
	    SWAP(data[i3],data[i3rev]);
	    SWAP(data[i3+1],data[i3rev+1]);
	  }
	}
      }
      ibit=ip2 >> 1;
      while (ibit >= ip1 && i2rev > ibit) {
	i2rev -= ibit;
	ibit >>= 1;
      }
      i2rev += ibit;
    }
    ifp1=ip1;
    while (ifp1 < ip2) {
      ifp2=ifp1 << 1;
      theta=isign*6.28318530717959/(ifp2/ip1);
      wtemp=sin(0.5*theta);
      wpr = -2.0*wtemp*wtemp;
      wpi=sin(theta);
      wr=1.0;
      wi=0.0;
      for (i3=1;i3<=ifp1;i3+=ip1) {
	for (i1=i3;i1<=i3+ip1-2;i1+=2) {
	  for (i2=i1;i2<=ip3;i2+=ifp2) {
	    k1=i2;
	    k2=k1+ifp1;
	    tempr=wr*data[k2]-wi*data[k2+1];
	    tempi=wr*data[k2+1]+wi*data[k2];
	    data[k2]=data[k1]-tempr;
	    data[k2+1]=data[k1+1]-tempi;
	    data[k1] += tempr;
	    data[k1+1] += tempi;
	  }
	}
	wr=(wtemp=wr)*wpr-wi*wpi+wr;
	wi=wi*wpr+wtemp*wpi+wi;
      }
      ifp1=ifp2;
    }
    nprev *= n;
  }
}
#undef SWAP


void rlft3(double ***data, double **speq, unsigned long nn1, unsigned long nn2,
	unsigned long nn3, int isign)
{
  void fourn(double data[], unsigned long nn[], int ndim, int isign);
  //void nrerror(char error_text[]);
  unsigned long i1,i2,i3,j1,j2,j3,nn[4],ii3;
  double theta,wi,wpi,wpr,wr,wtemp;
  double c1,c2,h1r,h1i,h2r,h2i;
  
  //if (1+&data[nn1][nn2][nn3]-&data[1][1][1] != nn1*nn2*nn3)
  //nrerror("rlft3: problem with dimensions or contiguity of data array\n");
  c1=0.5;
  c2 = -0.5*isign;
  theta=isign*(6.28318530717959/nn3);
  wtemp=sin(0.5*theta);
  wpr = -2.0*wtemp*wtemp;
  wpi=sin(theta);
  nn[1]=nn1;
  nn[2]=nn2;
  nn[3]=nn3 >> 1;
  if (isign == 1) {
    fourn(&data[1][1][1]-1,nn,3,isign);
    for (i1=1;i1<=nn1;i1++)
      for (i2=1,j2=0;i2<=nn2;i2++) {
	speq[i1][++j2]=data[i1][i2][1];
	speq[i1][++j2]=data[i1][i2][2];
      }
  }
  for (i1=1;i1<=nn1;i1++) {
    j1=(i1 != 1 ? nn1-i1+2 : 1);
    wr=1.0;
    wi=0.0;
    for (ii3=1,i3=1;i3<=(nn3>>2)+1;i3++,ii3+=2) {
      for (i2=1;i2<=nn2;i2++) {
	if (i3 == 1) {
	  j2=(i2 != 1 ? ((nn2-i2)<<1)+3 : 1);
	  h1r=c1*(data[i1][i2][1]+speq[j1][j2]);
	  h1i=c1*(data[i1][i2][2]-speq[j1][j2+1]);
	  h2i=c2*(data[i1][i2][1]-speq[j1][j2]);
	  h2r= -c2*(data[i1][i2][2]+speq[j1][j2+1]);
	  data[i1][i2][1]=h1r+h2r;
	  data[i1][i2][2]=h1i+h2i;
	  speq[j1][j2]=h1r-h2r;
	  speq[j1][j2+1]=h2i-h1i;
	} else {
	  j2=(i2 != 1 ? nn2-i2+2 : 1);
	  j3=nn3+3-(i3<<1);
	  h1r=c1*(data[i1][i2][ii3]+data[j1][j2][j3]);
	  h1i=c1*(data[i1][i2][ii3+1]-data[j1][j2][j3+1]);
	  h2i=c2*(data[i1][i2][ii3]-data[j1][j2][j3]);
	  h2r= -c2*(data[i1][i2][ii3+1]+data[j1][j2][j3+1]);
	  data[i1][i2][ii3]=h1r+wr*h2r-wi*h2i;
	  data[i1][i2][ii3+1]=h1i+wr*h2i+wi*h2r;
	  data[j1][j2][j3]=h1r-wr*h2r+wi*h2i;
	  data[j1][j2][j3+1]= -h1i+wr*h2i+wi*h2r;
	}
      }
      wr=(wtemp=wr)*wpr-wi*wpi+wr;
      wi=wi*wpr+wtemp*wpi+wi;
    }
  }
  if (isign == -1)
    fourn(&data[1][1][1]-1,nn,3,isign);
}


