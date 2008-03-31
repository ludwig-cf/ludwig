#ifndef LCPARALLEL_CC
#define LCPARALLEL_CC

#include "LCParallel.hh"
#include "mpi.h"
  
void exchangeMomentumAndQTensor()
{
  int ix,iy,iz;

  // --------------------
  // Sends densities
  ix=1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=density[ix][iy][iz];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD);
  ix=Lx2-2;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=density[ix][iy][iz];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD);

  // --------------------
  // Sends Qxx
  ix=1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=Qxx[ix][iy][iz];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD);
  ix=Lx2-2;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=Qxx[ix][iy][iz];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD);

  // --------------------
  // Sends Qxy
  ix=1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=Qxy[ix][iy][iz];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD);
  ix=Lx2-2;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=Qxy[ix][iy][iz];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD);

  // --------------------
  // Sends Qyy
  ix=1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=Qyy[ix][iy][iz];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD);
  ix=Lx2-2;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=Qyy[ix][iy][iz];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD);

  // --------------------
  // Sends Qxz
  ix=1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=Qxz[ix][iy][iz];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD);
  ix=Lx2-2;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=Qxz[ix][iy][iz];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD);

  // --------------------
  // Sends Qyz
  ix=1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=Qyz[ix][iy][iz];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD);
  ix=Lx2-2;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=Qyz[ix][iy][iz];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD);

  // --------------------
  // Sends ux
  ix=1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=u[ix][iy][iz][0];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD);
  ix=Lx2-2;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=u[ix][iy][iz][0];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD);

  // --------------------
  // Sends uy
  ix=1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=u[ix][iy][iz][1];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD);
  ix=Lx2-2;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=u[ix][iy][iz][1];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD);

  // --------------------
  // Sends uz
  ix=1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=u[ix][iy][iz][2];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD);
  ix=Lx2-2;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      tmpBuf[iy*Lz+iz]=u[ix][iy][iz][2];
  MPI_Bsend(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD);



  // --------------------
  // Receives densities
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD,&status);
  ix=Lx2-1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      density[ix][iy][iz]=tmpBuf[iy*Lz+iz];
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD,&status);
  ix=0;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      density[ix][iy][iz]=tmpBuf[iy*Lz+iz];
  // --------------------
  // Receives Qxx
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD,&status);
  ix=Lx2-1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      Qxx[ix][iy][iz]=tmpBuf[iy*Lz+iz];
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD,&status);
  ix=0;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      Qxx[ix][iy][iz]=tmpBuf[iy*Lz+iz];
  // --------------------
  // Receives Qxy
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD,&status);
  ix=Lx2-1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      Qxy[ix][iy][iz]=tmpBuf[iy*Lz+iz];
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD,&status);
  ix=0;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      Qxy[ix][iy][iz]=tmpBuf[iy*Lz+iz];
  // --------------------
  // Receives Qyy
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD,&status);
  ix=Lx2-1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      Qyy[ix][iy][iz]=tmpBuf[iy*Lz+iz];
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD,&status);
  ix=0;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      Qyy[ix][iy][iz]=tmpBuf[iy*Lz+iz];
  // --------------------
  // Receives Qxz
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD,&status);
  ix=Lx2-1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      Qxz[ix][iy][iz]=tmpBuf[iy*Lz+iz];
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD,&status);
  ix=0;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      Qxz[ix][iy][iz]=tmpBuf[iy*Lz+iz];
  // --------------------
  // Receives Qyz
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD,&status);
  ix=Lx2-1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      Qyz[ix][iy][iz]=tmpBuf[iy*Lz+iz];
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD,&status);
  ix=0;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      Qyz[ix][iy][iz]=tmpBuf[iy*Lz+iz];

  // --------------------
  // Receives ux
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD,&status);
  ix=Lx2-1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      u[ix][iy][iz][0]=tmpBuf[iy*Lz+iz];
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD,&status);
  ix=0;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      u[ix][iy][iz][0]=tmpBuf[iy*Lz+iz];
  // --------------------
  // Receives uy
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD,&status);
  ix=Lx2-1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      u[ix][iy][iz][1]=tmpBuf[iy*Lz+iz];
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD,&status);
  ix=0;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      u[ix][iy][iz][1]=tmpBuf[iy*Lz+iz];
  // --------------------
  // Receives uz
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD,&status);
  ix=Lx2-1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      u[ix][iy][iz][2]=tmpBuf[iy*Lz+iz];
  MPI_Recv(tmpBuf,Lz*Ly,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD,&status);
  ix=0;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++)
      u[ix][iy][iz][2]=tmpBuf[iy*Lz+iz];


//  MPI_Buffer_detach(&com, &buffer_size);
//  free(com);

}

void communicateOldDistributions(double ****fold)
{
  int ix,iy,iz;

  // --------------------
  // Sends fold
  ix=1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++) {
      tmpBuf[iy*Lz*5+iz*5]=fold[ix][iy][iz][3];
      tmpBuf[iy*Lz*5+iz*5+1]=fold[ix][iy][iz][8];
      tmpBuf[iy*Lz*5+iz*5+2]=fold[ix][iy][iz][9];
      tmpBuf[iy*Lz*5+iz*5+3]=fold[ix][iy][iz][12];
      tmpBuf[iy*Lz*5+iz*5+4]=fold[ix][iy][iz][13];
    }
  MPI_Bsend(tmpBuf,Lz*Ly*5,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD);
  ix=Lx2-2;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++) {
      tmpBuf[iy*Lz*5+iz*5]=fold[ix][iy][iz][1];
      tmpBuf[iy*Lz*5+iz*5+1]=fold[ix][iy][iz][7];
      tmpBuf[iy*Lz*5+iz*5+2]=fold[ix][iy][iz][10];
      tmpBuf[iy*Lz*5+iz*5+3]=fold[ix][iy][iz][11];
      tmpBuf[iy*Lz*5+iz*5+4]=fold[ix][iy][iz][14];
    }
  MPI_Bsend(tmpBuf,Lz*Ly*5,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD);


  // ========================================

  // --------------------
  // Receives fold
  MPI_Recv(tmpBuf,Lz*Ly*5,MPI_DOUBLE,rightNeighbor,0,MPI_COMM_WORLD,&status);
  ix=Lx2-1;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++) {
      fold[ix][iy][iz][3]=tmpBuf[iy*Lz*5+iz*5];
      fold[ix][iy][iz][8]=tmpBuf[iy*Lz*5+iz*5+1];
      fold[ix][iy][iz][9]=tmpBuf[iy*Lz*5+iz*5+2];
      fold[ix][iy][iz][12]=tmpBuf[iy*Lz*5+iz*5+3];
      fold[ix][iy][iz][13]=tmpBuf[iy*Lz*5+iz*5+4];
    }
  MPI_Recv(tmpBuf,Lz*Ly*5,MPI_DOUBLE,leftNeighbor,0,MPI_COMM_WORLD,&status);
  ix=0;
  for (iy=0;iy<Ly;iy++)
    for (iz=0;iz<Lz;iz++) {
      fold[ix][iy][iz][1]=tmpBuf[iy*Lz*5+iz*5];
      fold[ix][iy][iz][7]=tmpBuf[iy*Lz*5+iz*5+1];
      fold[ix][iy][iz][10]=tmpBuf[iy*Lz*5+iz*5+2];
      fold[ix][iy][iz][11]=tmpBuf[iy*Lz*5+iz*5+3];
      fold[ix][iy][iz][14]=tmpBuf[iy*Lz*5+iz*5+4];
    }

}

#endif
