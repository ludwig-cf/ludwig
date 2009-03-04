
#include "LCParallel.hh"

// Parallel, and then serial, code to deal with the exchange of
// information at the periodic and processor boundaries.

#ifdef PARALLEL

/****************************************************************************
 *
 *  exchangeMomentumAndQTensor
 *
 *  Three-dimensional communication
 *
 ****************************************************************************/ 

void exchangeMomentumAndQTensor() {

    extern double total_exch_;
    extern int pe_cartesian_neighbour_[2][3];
    extern MPI_Comm cartesian_communicator_;
    extern int Lx2, Ly2, Lz2;
    extern double *** density;
    extern double *** Qxx;
    extern double *** Qxy;
    extern double *** Qxz;
    extern double *** Qyy;
    extern double *** Qyz;
    extern double **** u;

    const int nquantity = 9;  /* density, Qxx, Qxy etc is 9 items */
    double * buf_sendforw;    /* send data to 'forward' direction */
    double * buf_sendback;    /* send data to 'backward' direction */
    double * buf_recvforw;    /* receive data from 'forward' direction */
    double * buf_recvback;    /* receive data from 'backward' direction */
    double   t0, t1;

    int ncount;
    int ix, iy, iz;
    int n, nforw, nback;

    const int tagb = 1002;
    const int tagf = 1003;

    MPI_Request send_request[2];
    MPI_Request recv_request[2];
    MPI_Status  send_status[2];
    MPI_Status  recv_status[2];
    MPI_Comm comm = cartesian_communicator_;

    t0 = MPI_Wtime();

    /* allocate buffers */

    ncount = nquantity*Ly2*Lz2;
    buf_sendforw = new double[ncount];
    buf_sendback = new double[ncount];
    buf_recvforw = new double[ncount];
    buf_recvback = new double[ncount];

    /* X DIRECTION */


    nforw = pe_cartesian_neighbour_[1][0];
    nback = pe_cartesian_neighbour_[0][0];

    /* post receives */

    MPI_Irecv(buf_recvforw, ncount, MPI_DOUBLE, nforw, tagb, comm,
              recv_request);
    MPI_Irecv(buf_recvback, ncount, MPI_DOUBLE, nback, tagf, comm,
              recv_request+1);

    /* load send buffers and non-blocking sends */

    ix = 1;
    n  = 0;

    for (iy = 0; iy < Ly2; iy++) {
        for (iz = 0; iz < Lz2; iz++) {
            buf_sendback[n++] = density[ix][iy][iz];
            buf_sendback[n++] =     Qxx[ix][iy][iz];
            buf_sendback[n++] =     Qxy[ix][iy][iz];
            buf_sendback[n++] =     Qyy[ix][iy][iz];
            buf_sendback[n++] =     Qxz[ix][iy][iz];
            buf_sendback[n++] =     Qyz[ix][iy][iz];
            buf_sendback[n++] =       u[ix][iy][iz][0];
            buf_sendback[n++] =       u[ix][iy][iz][1];
            buf_sendback[n++] =       u[ix][iy][iz][2];
        }
    }

    MPI_Issend(buf_sendback, ncount, MPI_DOUBLE, nback, tagb, comm,
               send_request);

    ix = Lx2-2;
    n  = 0;

    for (iy = 0; iy < Ly2; iy++) {
        for (iz = 0; iz < Lz2; iz++) {
            buf_sendforw[n++] = density[ix][iy][iz];
            buf_sendforw[n++] =     Qxx[ix][iy][iz];
            buf_sendforw[n++] =     Qxy[ix][iy][iz];
            buf_sendforw[n++] =     Qyy[ix][iy][iz];
            buf_sendforw[n++] =     Qxz[ix][iy][iz];
            buf_sendforw[n++] =     Qyz[ix][iy][iz];
            buf_sendforw[n++] =       u[ix][iy][iz][0];
            buf_sendforw[n++] =       u[ix][iy][iz][1];
            buf_sendforw[n++] =       u[ix][iy][iz][2];
        }
    }

    MPI_Issend(buf_sendforw, ncount, MPI_DOUBLE, nforw, tagf, comm,
               send_request+1);

    /* wait for receives to complete and unload buffers */

    MPI_Waitall(2, recv_request, recv_status);

    ix = Lx2-1;
    n  = 0;

    for (iy = 0; iy < Ly2; iy++) {
        for (iz = 0; iz < Lz2; iz++) {
            density[ix][iy][iz] = buf_recvforw[n++];
            Qxx[ix][iy][iz]  = buf_recvforw[n++];
            Qxy[ix][iy][iz]  = buf_recvforw[n++];
            Qyy[ix][iy][iz]  = buf_recvforw[n++];
            Qxz[ix][iy][iz]  = buf_recvforw[n++];
            Qyz[ix][iy][iz]  = buf_recvforw[n++];
            u[ix][iy][iz][0] = buf_recvforw[n++];
            u[ix][iy][iz][1] = buf_recvforw[n++];
            u[ix][iy][iz][2] = buf_recvforw[n++];
        }
    }

    ix = 0;
    n  = 0;

    for (iy = 0; iy < Ly2; iy++) {
        for (iz = 0; iz < Lz2; iz++) {
            density[ix][iy][iz] = buf_recvback[n++];
            Qxx[ix][iy][iz]  = buf_recvback[n++];
            Qxy[ix][iy][iz]  = buf_recvback[n++];
            Qyy[ix][iy][iz]  = buf_recvback[n++];
            Qxz[ix][iy][iz]  = buf_recvback[n++];
            Qyz[ix][iy][iz]  = buf_recvback[n++];
            u[ix][iy][iz][0] = buf_recvback[n++];
            u[ix][iy][iz][1] = buf_recvback[n++];
            u[ix][iy][iz][2] = buf_recvback[n++];
        }
    }

    /* mop up the sends */

    MPI_Waitall(2, send_request, send_status);

    delete buf_sendforw;
    delete buf_sendback;
    delete buf_recvforw;
    delete buf_recvback;

    /* REPEAT FOR Y */

    /* allocate buffers */

    ncount = nquantity*Lx2*Lz2;
    buf_sendforw = new double[ncount];
    buf_sendback = new double[ncount];
    buf_recvforw = new double[ncount];
    buf_recvback = new double[ncount];


    nforw = pe_cartesian_neighbour_[1][1];
    nback = pe_cartesian_neighbour_[0][1];

    /* post receives */

    MPI_Irecv(buf_recvforw, ncount, MPI_DOUBLE, nforw, tagb, comm,
              recv_request);
    MPI_Irecv(buf_recvback, ncount, MPI_DOUBLE, nback, tagf, comm,
              recv_request+1);

    /* load send buffers and non-blocking sends */

    iy = 1;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iz = 0; iz < Lz2; iz++) {
            buf_sendback[n++] = density[ix][iy][iz];
            buf_sendback[n++] =     Qxx[ix][iy][iz];
            buf_sendback[n++] =     Qxy[ix][iy][iz];
            buf_sendback[n++] =     Qyy[ix][iy][iz];
            buf_sendback[n++] =     Qxz[ix][iy][iz];
            buf_sendback[n++] =     Qyz[ix][iy][iz];
            buf_sendback[n++] =       u[ix][iy][iz][0];
            buf_sendback[n++] =       u[ix][iy][iz][1];
            buf_sendback[n++] =       u[ix][iy][iz][2];
        }
    }

    MPI_Issend(buf_sendback, ncount, MPI_DOUBLE, nback, tagb, comm,
               send_request);

    iy = Ly2-2;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iz = 0; iz < Lz2; iz++) {
            buf_sendforw[n++] = density[ix][iy][iz];
            buf_sendforw[n++] =     Qxx[ix][iy][iz];
            buf_sendforw[n++] =     Qxy[ix][iy][iz];
            buf_sendforw[n++] =     Qyy[ix][iy][iz];
            buf_sendforw[n++] =     Qxz[ix][iy][iz];
            buf_sendforw[n++] =     Qyz[ix][iy][iz];
            buf_sendforw[n++] =       u[ix][iy][iz][0];
            buf_sendforw[n++] =       u[ix][iy][iz][1];
            buf_sendforw[n++] =       u[ix][iy][iz][2];
        }
    }

    MPI_Issend(buf_sendforw, ncount, MPI_DOUBLE, nforw, tagf, comm,
               send_request+1);

    /* wait for receives to complete and unload buffers */

    MPI_Waitall(2, recv_request, recv_status);

    iy = Ly2-1;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iz = 0; iz < Lz2; iz++) {
            density[ix][iy][iz] = buf_recvforw[n++];
            Qxx[ix][iy][iz]  = buf_recvforw[n++];
            Qxy[ix][iy][iz]  = buf_recvforw[n++];
            Qyy[ix][iy][iz]  = buf_recvforw[n++];
            Qxz[ix][iy][iz]  = buf_recvforw[n++];
            Qyz[ix][iy][iz]  = buf_recvforw[n++];
            u[ix][iy][iz][0] = buf_recvforw[n++];
            u[ix][iy][iz][1] = buf_recvforw[n++];
            u[ix][iy][iz][2] = buf_recvforw[n++];
        }
    }

    iy = 0;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iz = 0; iz < Lz2; iz++) {
            density[ix][iy][iz] = buf_recvback[n++];
            Qxx[ix][iy][iz]  = buf_recvback[n++];
            Qxy[ix][iy][iz]  = buf_recvback[n++];
            Qyy[ix][iy][iz]  = buf_recvback[n++];
            Qxz[ix][iy][iz]  = buf_recvback[n++];
            Qyz[ix][iy][iz]  = buf_recvback[n++];
            u[ix][iy][iz][0] = buf_recvback[n++];
            u[ix][iy][iz][1] = buf_recvback[n++];
            u[ix][iy][iz][2] = buf_recvback[n++];
        }
    }

    /* mop up the sends */

    MPI_Waitall(2, send_request, send_status);

    delete buf_sendforw;
    delete buf_sendback;
    delete buf_recvforw;
    delete buf_recvback;

    /* REPEAT FOR Z */

    /* allocate buffers */

    ncount = nquantity*Lx2*Ly2;
    buf_sendforw = new double[ncount];
    buf_sendback = new double[ncount];
    buf_recvforw = new double[ncount];
    buf_recvback = new double[ncount];

    nforw = pe_cartesian_neighbour_[1][2];
    nback = pe_cartesian_neighbour_[0][2];

    /* post receives */

    MPI_Irecv(buf_recvforw, ncount, MPI_DOUBLE, nforw, tagb, comm,
              recv_request);
    MPI_Irecv(buf_recvback, ncount, MPI_DOUBLE, nback, tagf, comm,
              recv_request+1);

    /* load send buffers and non-blocking sends */

    iz = 1;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iy = 0; iy < Ly2; iy++) {
            buf_sendback[n++] = density[ix][iy][iz];
            buf_sendback[n++] =     Qxx[ix][iy][iz];
            buf_sendback[n++] =     Qxy[ix][iy][iz];
            buf_sendback[n++] =     Qyy[ix][iy][iz];
            buf_sendback[n++] =     Qxz[ix][iy][iz];
            buf_sendback[n++] =     Qyz[ix][iy][iz];
            buf_sendback[n++] =       u[ix][iy][iz][0];
            buf_sendback[n++] =       u[ix][iy][iz][1];
            buf_sendback[n++] =       u[ix][iy][iz][2];
        }
    }

    MPI_Issend(buf_sendback, ncount, MPI_DOUBLE, nback, tagb, comm,
               send_request);

    iz = Lz2-2;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iy = 0; iy < Ly2; iy++) {
            buf_sendforw[n++] = density[ix][iy][iz];
            buf_sendforw[n++] =     Qxx[ix][iy][iz];
            buf_sendforw[n++] =     Qxy[ix][iy][iz];
            buf_sendforw[n++] =     Qyy[ix][iy][iz];
            buf_sendforw[n++] =     Qxz[ix][iy][iz];
            buf_sendforw[n++] =     Qyz[ix][iy][iz];
            buf_sendforw[n++] =       u[ix][iy][iz][0];
            buf_sendforw[n++] =       u[ix][iy][iz][1];
            buf_sendforw[n++] =       u[ix][iy][iz][2];
        }
    }

    MPI_Issend(buf_sendforw, ncount, MPI_DOUBLE, nforw, tagf, comm,
               send_request+1);

    /* wait for receives to complete and unload buffers */

    MPI_Waitall(2, recv_request, recv_status);

    iz = Lz2-1;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iy = 0; iy < Ly2; iy++) {
            density[ix][iy][iz] = buf_recvforw[n++];
            Qxx[ix][iy][iz]  = buf_recvforw[n++];
            Qxy[ix][iy][iz]  = buf_recvforw[n++];
            Qyy[ix][iy][iz]  = buf_recvforw[n++];
            Qxz[ix][iy][iz]  = buf_recvforw[n++];
            Qyz[ix][iy][iz]  = buf_recvforw[n++];
            u[ix][iy][iz][0] = buf_recvforw[n++];
            u[ix][iy][iz][1] = buf_recvforw[n++];
            u[ix][iy][iz][2] = buf_recvforw[n++];
        }
    }

    iz = 0;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iy = 0; iy < Ly2; iy++) {
            density[ix][iy][iz] = buf_recvback[n++];
            Qxx[ix][iy][iz]  = buf_recvback[n++];
            Qxy[ix][iy][iz]  = buf_recvback[n++];
            Qyy[ix][iy][iz]  = buf_recvback[n++];
            Qxz[ix][iy][iz]  = buf_recvback[n++];
            Qyz[ix][iy][iz]  = buf_recvback[n++];
            u[ix][iy][iz][0] = buf_recvback[n++];
            u[ix][iy][iz][1] = buf_recvback[n++];
            u[ix][iy][iz][2] = buf_recvback[n++];
        }
    }

    /* mop up the sends */

    MPI_Waitall(2, send_request, send_status);

    delete buf_sendforw;
    delete buf_sendback;
    delete buf_recvforw;
    delete buf_recvback;

    t1 = MPI_Wtime();
    total_exch_ += (t1-t0);

    return;
}

/*****************************************************************************
 *
 *  communicateOldDistributions
 *
 *****************************************************************************/

void communicateOldDistributions(double **** fold) {

    extern double total_comm_;
    extern int pe_cartesian_neighbour_[2][3];
    extern MPI_Comm cartesian_communicator_;
    extern int Lx2, Ly2, Lz2;

    const int nquantity = 5;   /* 5 propagating distributions */
    double * buf_sendforw;     /* send data to 'forward' direction */
    double * buf_sendback;     /* send data to 'backward' direction */
    double * buf_recvforw;     /* receive data from 'forward' direction */
    double * buf_recvback;     /* receive data from 'backward' direction */
    double   t0, t1;
    //extern double total_comm_;

    int ncount;
    int ix, iy, iz;
    int n, nforw, nback;

    const int tagb = 1014;
    const int tagf = 1015;

    MPI_Request send_request[2];
    MPI_Request recv_request[2];
    MPI_Status  send_status[2];
    MPI_Status  recv_status[2];
    MPI_Comm    comm = cartesian_communicator_;

    t0 = MPI_Wtime();

    /* X DIRECTION */
    /* Allocate buffers */

    ncount = nquantity*Ly2*Lz2;
    buf_sendforw = new double[ncount];
    buf_sendback = new double[ncount];
    buf_recvforw = new double[ncount];
    buf_recvback = new double[ncount];

    nforw = pe_cartesian_neighbour_[1][0];
    nback = pe_cartesian_neighbour_[0][0];

    /* post receives */

    MPI_Irecv(buf_recvforw, ncount, MPI_DOUBLE, nforw, tagb, comm,
              recv_request);
    MPI_Irecv(buf_recvback, ncount, MPI_DOUBLE, nback, tagf, comm,
              recv_request+1);

    /* load send buffers and non-blocking sends */

    ix = 1;
    n  = 0;

    for (iy = 0; iy < Ly2; iy++) {
        for (iz = 0; iz < Lz2; iz++) {
            buf_sendback[n++] = fold[ix][iy][iz][3];
            buf_sendback[n++] = fold[ix][iy][iz][8];
            buf_sendback[n++] = fold[ix][iy][iz][9];
            buf_sendback[n++] = fold[ix][iy][iz][12];
            buf_sendback[n++] = fold[ix][iy][iz][13];
        }
    }

    MPI_Issend(buf_sendback, ncount, MPI_DOUBLE, nback, tagb, comm,
               send_request);


    ix = Lx2-2;
    n  = 0;

    for (iy = 0; iy < Ly2; iy++) {
        for (iz = 0; iz < Lz2; iz++) {
            buf_sendforw[n++] = fold[ix][iy][iz][1];
            buf_sendforw[n++] = fold[ix][iy][iz][7];
            buf_sendforw[n++] = fold[ix][iy][iz][10];
            buf_sendforw[n++] = fold[ix][iy][iz][11];
            buf_sendforw[n++] = fold[ix][iy][iz][14];
        }
    }

    MPI_Issend(buf_sendforw, ncount, MPI_DOUBLE, nforw, tagf, comm,
               send_request+1);

    /* wait for receives to complete and unload buffers */

    MPI_Waitall(2, recv_request, recv_status);

    ix = Lx2-1;
    n  = 0;

    for (iy = 0; iy < Ly2; iy++) {
        for (iz = 0; iz < Lz2; iz++) {
            fold[ix][iy][iz][3]  = buf_recvforw[n++];
            fold[ix][iy][iz][8]  = buf_recvforw[n++];
            fold[ix][iy][iz][9]  = buf_recvforw[n++];
            fold[ix][iy][iz][12] = buf_recvforw[n++];
            fold[ix][iy][iz][13] = buf_recvforw[n++];
        }
    }

    ix = 0;
    n  = 0;

    for (iy = 0; iy < Ly2; iy++) {
        for (iz = 0; iz < Lz2; iz++) {
            fold[ix][iy][iz][1]  = buf_recvback[n++];
            fold[ix][iy][iz][7]  = buf_recvback[n++];
            fold[ix][iy][iz][10] = buf_recvback[n++];
            fold[ix][iy][iz][11] = buf_recvback[n++];
            fold[ix][iy][iz][14] = buf_recvback[n++];
        }
    }

    /* mop up the sends */

    MPI_Waitall(2, send_request, send_status);

    delete buf_sendforw;
    delete buf_sendback;
    delete buf_recvforw;
    delete buf_recvback;


    /* Y DIRECTION */
    /* Allocate buffers */

    ncount = nquantity*Lx2*Lz2;
    buf_sendforw = new double[ncount];
    buf_sendback = new double[ncount];
    buf_recvforw = new double[ncount];
    buf_recvback = new double[ncount];

    /* Y DIRECTION */

    nforw = pe_cartesian_neighbour_[1][1];
    nback = pe_cartesian_neighbour_[0][1];

    /* post receives */

    MPI_Irecv(buf_recvforw, ncount, MPI_DOUBLE, nforw, tagb, comm,
              recv_request);
    MPI_Irecv(buf_recvback, ncount, MPI_DOUBLE, nback, tagf, comm,
              recv_request+1);

    /* load send buffers and non-blocking sends */

    iy = 1;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iz = 0; iz < Lz2; iz++) {
            buf_sendback[n++] = fold[ix][iy][iz][4];
            buf_sendback[n++] = fold[ix][iy][iz][9];
            buf_sendback[n++] = fold[ix][iy][iz][10];
            buf_sendback[n++] = fold[ix][iy][iz][13];
            buf_sendback[n++] = fold[ix][iy][iz][14];
        }
    }

    MPI_Issend(buf_sendback, ncount, MPI_DOUBLE, nback, tagb, comm,
               send_request);


    iy = Ly2-2;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iz = 0; iz < Lz2; iz++) {
            buf_sendforw[n++] = fold[ix][iy][iz][2];
            buf_sendforw[n++] = fold[ix][iy][iz][7];
            buf_sendforw[n++] = fold[ix][iy][iz][8];
            buf_sendforw[n++] = fold[ix][iy][iz][11];
            buf_sendforw[n++] = fold[ix][iy][iz][12];
        }
    }

    MPI_Issend(buf_sendforw, ncount, MPI_DOUBLE, nforw, tagf, comm,
               send_request+1);

    /* wait for receives to complete and unload buffers */

    MPI_Waitall(2, recv_request, recv_status);

    iy = Ly2-1;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iz = 0; iz < Lz2; iz++) {
            fold[ix][iy][iz][4]  = buf_recvforw[n++];
            fold[ix][iy][iz][9]  = buf_recvforw[n++];
            fold[ix][iy][iz][10]  = buf_recvforw[n++];
            fold[ix][iy][iz][13] = buf_recvforw[n++];
            fold[ix][iy][iz][14] = buf_recvforw[n++];
        }
    }

    iy = 0;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iz = 0; iz < Lz2; iz++) {
            fold[ix][iy][iz][2]  = buf_recvback[n++];
            fold[ix][iy][iz][7]  = buf_recvback[n++];
            fold[ix][iy][iz][8] = buf_recvback[n++];
            fold[ix][iy][iz][11] = buf_recvback[n++];
            fold[ix][iy][iz][12] = buf_recvback[n++];
        }
    }

    /* mop up the sends */

    MPI_Waitall(2, send_request, send_status);

    delete buf_sendforw;
    delete buf_sendback;
    delete buf_recvforw;
    delete buf_recvback;


    /* Z DIRECTION */
    /* Allocate buffers */

    ncount = nquantity*Lx2*Ly2;
    buf_sendforw = new double[ncount];
    buf_sendback = new double[ncount];
    buf_recvforw = new double[ncount];
    buf_recvback = new double[ncount];

    nforw = pe_cartesian_neighbour_[1][2];
    nback = pe_cartesian_neighbour_[0][2];

    /* post receives */

    MPI_Irecv(buf_recvforw, ncount, MPI_DOUBLE, nforw, tagb, comm,
              recv_request);
    MPI_Irecv(buf_recvback, ncount, MPI_DOUBLE, nback, tagf, comm,
              recv_request+1);

    /* load send buffers and non-blocking sends */

    iz = 1;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iy = 0; iy < Ly2; iy++) {
            buf_sendback[n++] = fold[ix][iy][iz][6];
            buf_sendback[n++] = fold[ix][iy][iz][11];
            buf_sendback[n++] = fold[ix][iy][iz][12];
            buf_sendback[n++] = fold[ix][iy][iz][13];
            buf_sendback[n++] = fold[ix][iy][iz][14];
        }
    }

    MPI_Issend(buf_sendback, ncount, MPI_DOUBLE, nback, tagb, comm,
               send_request);


    iz = Lz2-2;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iy = 0; iy < Ly2; iy++) {
            buf_sendforw[n++] = fold[ix][iy][iz][5];
            buf_sendforw[n++] = fold[ix][iy][iz][7];
            buf_sendforw[n++] = fold[ix][iy][iz][8];
            buf_sendforw[n++] = fold[ix][iy][iz][9];
            buf_sendforw[n++] = fold[ix][iy][iz][10];
        }
    }

    MPI_Issend(buf_sendforw, ncount, MPI_DOUBLE, nforw, tagf, comm,
               send_request+1);

    /* wait for receives to complete and unload buffers */

    MPI_Waitall(2, recv_request, recv_status);

    iz = Lz2-1;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iy = 0; iy < Ly2; iy++) {
            fold[ix][iy][iz][6]  = buf_recvforw[n++];
            fold[ix][iy][iz][11]  = buf_recvforw[n++];
            fold[ix][iy][iz][12]  = buf_recvforw[n++];
            fold[ix][iy][iz][13] = buf_recvforw[n++];
            fold[ix][iy][iz][14] = buf_recvforw[n++];
        }
    }

    iz = 0;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iy = 0; iy < Ly2; iy++) {
            fold[ix][iy][iz][5]  = buf_recvback[n++];
            fold[ix][iy][iz][7]  = buf_recvback[n++];
            fold[ix][iy][iz][8] = buf_recvback[n++];
            fold[ix][iy][iz][9] = buf_recvback[n++];
            fold[ix][iy][iz][10] = buf_recvback[n++];
        }
    }

    /* mop up the sends */

    MPI_Waitall(2, send_request, send_status);

    delete buf_sendforw;
    delete buf_sendback;
    delete buf_recvforw;
    delete buf_recvback;

    t1 = MPI_Wtime();
    total_comm_ += (t1-t0);

    return;
}

/*****************************************************************************
 *
 *  exchangeTau
 *
 *****************************************************************************/

void exchangeTau() {

    extern double total_comm_;
    extern int pe_cartesian_neighbour_[2][3];
    extern MPI_Comm cartesian_communicator_;
    extern int Lx2, Ly2, Lz2;

    extern double *** tauxy;  /* X and Y exchange required */
    extern double *** tauxz;  /* X and Z exchange required */
    extern double *** tauyz;  /* Y and Z exchange required */

    const int nquantity = 2;   /* 2 components tau each direction */
    double * buf_sendforw;     /* send data to 'forward' direction */
    double * buf_sendback;     /* send data to 'backward' direction */
    double * buf_recvforw;     /* receive data from 'forward' direction */
    double * buf_recvback;     /* receive data from 'backward' direction */

    int ncount;
    int ix, iy, iz;
    int n, nforw, nback;

    const int tagb = 1016;
    const int tagf = 1017;

    MPI_Request send_request[2];
    MPI_Request recv_request[2];
    MPI_Status  send_status[2];
    MPI_Status  recv_status[2];
    MPI_Comm    comm = cartesian_communicator_;

    /* X direction - tauxy and tauxz */
    /* Allocate buffers */

    ncount = nquantity*Ly2*Lz2;
    buf_sendforw = new double[ncount];
    buf_sendback = new double[ncount];
    buf_recvforw = new double[ncount];
    buf_recvback = new double[ncount];

    nforw = pe_cartesian_neighbour_[1][0];
    nback = pe_cartesian_neighbour_[0][0];

    /* post receives */

    MPI_Irecv(buf_recvforw, ncount, MPI_DOUBLE, nforw, tagb, comm,
              recv_request);
    MPI_Irecv(buf_recvback, ncount, MPI_DOUBLE, nback, tagf, comm,
              recv_request+1);

    /* load send buffers and non-blocking sends */

    ix = 1;
    n  = 0;

    for (iy = 0; iy < Ly2; iy++) {
        for (iz = 0; iz < Lz2; iz++) {
            buf_sendback[n++] = tauxy[ix][iy][iz];
            buf_sendback[n++] = tauxz[ix][iy][iz];
        }
    }

    MPI_Issend(buf_sendback, ncount, MPI_DOUBLE, nback, tagb, comm,
               send_request);


    ix = Lx2-2;
    n  = 0;

    for (iy = 0; iy < Ly2; iy++) {
        for (iz = 0; iz < Lz2; iz++) {
            buf_sendforw[n++] = tauxy[ix][iy][iz];
            buf_sendforw[n++] = tauxz[ix][iy][iz];
        }
    }

    MPI_Issend(buf_sendforw, ncount, MPI_DOUBLE, nforw, tagf, comm,
               send_request+1);

    /* wait for receives to complete and unload buffers */

    MPI_Waitall(2, recv_request, recv_status);

    ix = Lx2-1;
    n  = 0;

    for (iy = 0; iy < Ly2; iy++) {
        for (iz = 0; iz < Lz2; iz++) {
            tauxy[ix][iy][iz]  = buf_recvforw[n++];
            tauxz[ix][iy][iz]  = buf_recvforw[n++];
        }
    }

    ix = 0;
    n  = 0;

    for (iy = 0; iy < Ly2; iy++) {
        for (iz = 0; iz < Lz2; iz++) {
            tauxy[ix][iy][iz]  = buf_recvback[n++];
            tauxz[ix][iy][iz]  = buf_recvback[n++];
        }
    }

    /* mop up the sends */

    MPI_Waitall(2, send_request, send_status);

    delete buf_sendforw;
    delete buf_sendback;
    delete buf_recvforw;
    delete buf_recvback;


    /* Y DIRECTION - tauxy and tauyz */
    /* Allocate buffers */

    ncount = nquantity*Lx2*Lz2;
    buf_sendforw = new double[ncount];
    buf_sendback = new double[ncount];
    buf_recvforw = new double[ncount];
    buf_recvback = new double[ncount];

    nforw = pe_cartesian_neighbour_[1][1];
    nback = pe_cartesian_neighbour_[0][1];

    /* post receives */

    MPI_Irecv(buf_recvforw, ncount, MPI_DOUBLE, nforw, tagb, comm,
              recv_request);
    MPI_Irecv(buf_recvback, ncount, MPI_DOUBLE, nback, tagf, comm,
              recv_request+1);

    /* load send buffers and non-blocking sends */

    iy = 1;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iz = 0; iz < Lz2; iz++) {
            buf_sendback[n++] = tauxy[ix][iy][iz];
            buf_sendback[n++] = tauyz[ix][iy][iz];
        }
    }

    MPI_Issend(buf_sendback, ncount, MPI_DOUBLE, nback, tagb, comm,
               send_request);


    iy = Ly2-2;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iz = 0; iz < Lz2; iz++) {
            buf_sendforw[n++] = tauxy[ix][iy][iz];
            buf_sendforw[n++] = tauyz[ix][iy][iz];
        }
    }

    MPI_Issend(buf_sendforw, ncount, MPI_DOUBLE, nforw, tagf, comm,
               send_request+1);

    /* wait for receives to complete and unload buffers */

    MPI_Waitall(2, recv_request, recv_status);

    iy = Ly2-1;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iz = 0; iz < Lz2; iz++) {
            tauxy[ix][iy][iz]  = buf_recvforw[n++];
            tauyz[ix][iy][iz]  = buf_recvforw[n++];
        }
    }

    iy = 0;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iz = 0; iz < Lz2; iz++) {
            tauxy[ix][iy][iz]  = buf_recvback[n++];
            tauyz[ix][iy][iz]  = buf_recvback[n++];
        }
    }

    /* mop up the sends */

    MPI_Waitall(2, send_request, send_status);

    delete buf_sendforw;
    delete buf_sendback;
    delete buf_recvforw;
    delete buf_recvback;

    /* Z DIRECTION - tauxz and tauyz */
    /* Allocate buffers */

    ncount = nquantity*Lx2*Ly2;
    buf_sendforw = new double[ncount];
    buf_sendback = new double[ncount];
    buf_recvforw = new double[ncount];
    buf_recvback = new double[ncount];

    nforw = pe_cartesian_neighbour_[1][2];
    nback = pe_cartesian_neighbour_[0][2];

    /* post receives */

    MPI_Irecv(buf_recvforw, ncount, MPI_DOUBLE, nforw, tagb, comm,
              recv_request);
    MPI_Irecv(buf_recvback, ncount, MPI_DOUBLE, nback, tagf, comm,
              recv_request+1);

    /* load send buffers and non-blocking sends */

    iz = 1;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iy = 0; iy < Ly2; iy++) {
            buf_sendback[n++] = tauxz[ix][iy][iz];
            buf_sendback[n++] = tauyz[ix][iy][iz];
        }
    }

    MPI_Issend(buf_sendback, ncount, MPI_DOUBLE, nback, tagb, comm,
               send_request);


    iz = Lz2-2;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iy = 0; iy < Ly2; iy++) {
            buf_sendforw[n++] = tauxz[ix][iy][iz];
            buf_sendforw[n++] = tauyz[ix][iy][iz];
        }
    }

    MPI_Issend(buf_sendforw, ncount, MPI_DOUBLE, nforw, tagf, comm,
               send_request+1);

    /* wait for receives to complete and unload buffers */

    MPI_Waitall(2, recv_request, recv_status);

    iz = Lz2-1;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iy = 0; iy < Ly2; iy++) {
            tauxz[ix][iy][iz] = buf_recvforw[n++];
            tauyz[ix][iy][iz] = buf_recvforw[n++];
        }
    }

    iz = 0;
    n  = 0;

    for (ix = 0; ix < Lx2; ix++) {
        for (iy = 0; iy < Ly2; iy++) {
            tauxz[ix][iy][iz] = buf_recvback[n++];
            tauyz[ix][iy][iz] = buf_recvback[n++];
        }
    }

    /* mop up the sends */

    MPI_Waitall(2, send_request, send_status);

    delete buf_sendforw;
    delete buf_sendback;
    delete buf_recvforw;
    delete buf_recvback;


  return;
}

#else /* PARALLEL; below are serial equivalents */

/****************************************************************************
 *
 *  exchangeMomentumAndQTensor
 *
 *  Serial version: copy density, five components of Q, and three of u.
 *
 ****************************************************************************/

void exchangeMomentumAndQTensor() {

  extern int Lx2, Ly2, Lz2;
  extern double *** density;
  extern double *** Qxx;
  extern double *** Qxy;
  extern double *** Qxz;
  extern double *** Qyy;
  extern double *** Qyz;
  extern double **** u;

  int ix, iy, iz;

  /* X DIRECTION */

  for (iy = 0; iy < Ly2; iy++) {
    for (iz = 0; iz < Lz2; iz++) {
      density[Lx2-1][iy][iz] = density[1][iy][iz];
          Qxx[Lx2-1][iy][iz]     = Qxx[1][iy][iz];
          Qxy[Lx2-1][iy][iz]     = Qxy[1][iy][iz];
          Qyy[Lx2-1][iy][iz]     = Qyy[1][iy][iz];
          Qxz[Lx2-1][iy][iz]     = Qxz[1][iy][iz];
          Qyz[Lx2-1][iy][iz]     = Qyz[1][iy][iz];
            u[Lx2-1][iy][iz][0]  =   u[1][iy][iz][0];
            u[Lx2-1][iy][iz][1]  =   u[1][iy][iz][1];
            u[Lx2-1][iy][iz][2]  =   u[1][iy][iz][2];

      density[0][iy][iz] = density[Lx2-2][iy][iz];
          Qxx[0][iy][iz]     = Qxx[Lx2-2][iy][iz];
          Qxy[0][iy][iz]     = Qxy[Lx2-2][iy][iz];
          Qyy[0][iy][iz]     = Qyy[Lx2-2][iy][iz];
          Qxz[0][iy][iz]     = Qxz[Lx2-2][iy][iz];
          Qyz[0][iy][iz]     = Qyz[Lx2-2][iy][iz];
            u[0][iy][iz][0]  =   u[Lx2-2][iy][iz][0];
            u[0][iy][iz][1]  =   u[Lx2-2][iy][iz][1];
            u[0][iy][iz][2]  =   u[Lx2-2][iy][iz][2];
    }
  }

  /* Y DIRECTION */

  for (ix = 0; ix < Lx2; ix++) {
    for (iz = 0; iz < Lz2; iz++) {
      density[ix][Ly2-1][iz] = density[ix][1][iz];
          Qxx[ix][Ly2-1][iz]    =  Qxx[ix][1][iz];
          Qxy[ix][Ly2-1][iz]    =  Qxy[ix][1][iz];
          Qyy[ix][Ly2-1][iz]    =  Qyy[ix][1][iz];
          Qxz[ix][Ly2-1][iz]    =  Qxz[ix][1][iz];
          Qyz[ix][Ly2-1][iz]    =  Qyz[ix][1][iz];
            u[ix][Ly2-1][iz][0] =    u[ix][1][iz][0];
            u[ix][Ly2-1][iz][1] =    u[ix][1][iz][1];
            u[ix][Ly2-1][iz][2] =    u[ix][1][iz][2];

      density[ix][0][iz] = density[ix][Ly2-2][iz];
          Qxx[ix][0][iz]     = Qxx[ix][Ly2-2][iz];
          Qxy[ix][0][iz]     = Qxy[ix][Ly2-2][iz];
          Qyy[ix][0][iz]     = Qyy[ix][Ly2-2][iz];
          Qxz[ix][0][iz]     = Qxz[ix][Ly2-2][iz];
          Qyz[ix][0][iz]     = Qyz[ix][Ly2-2][iz];
            u[ix][0][iz][0]  =   u[ix][Ly2-2][iz][0];
            u[ix][0][iz][1]  =   u[ix][Ly2-2][iz][1];
            u[ix][0][iz][2]  =   u[ix][Ly2-2][iz][2];
    }
  }

  /* Z DIRECTION */

  for (ix = 0; ix < Lx2; ix++) {
    for (iy = 0; iy < Ly2; iy++) {
      density[ix][iy][Lz2-1] = density[ix][iy][1];
          Qxx[ix][iy][Lz2-1]    =  Qxx[ix][iy][1];
          Qxy[ix][iy][Lz2-1]    =  Qxy[ix][iy][1];
          Qyy[ix][iy][Lz2-1]    =  Qyy[ix][iy][1];
          Qxz[ix][iy][Lz2-1]    =  Qxz[ix][iy][1];
          Qyz[ix][iy][Lz2-1]    =  Qyz[ix][iy][1];
            u[ix][iy][Lz2-1][0] =    u[ix][iy][1][0];
            u[ix][iy][Lz2-1][1] =    u[ix][iy][1][1];
            u[ix][iy][Lz2-1][2] =    u[ix][iy][1][2];

      density[ix][iy][0] = density[ix][iy][Lz2-2];
          Qxx[ix][iy][0]     = Qxx[ix][iy][Lz2-2];
          Qxy[ix][iy][0]     = Qxy[ix][iy][Lz2-2];
          Qyy[ix][iy][0]     = Qyy[ix][iy][Lz2-2];
          Qxz[ix][iy][0]     = Qxz[ix][iy][Lz2-2];
          Qyz[ix][iy][0]     = Qyz[ix][iy][Lz2-2];
            u[ix][iy][0][0]  =   u[ix][iy][Lz2-2][0];
            u[ix][iy][0][1]  =   u[ix][iy][Lz2-2][1];
            u[ix][iy][0][2]  =   u[ix][iy][Lz2-2][2];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  communicateOldDistributions
 *
 *****************************************************************************/

void communicateOldDistributions(double **** fold) {

  extern int Lx2, Ly2, Lz2;
  int ix, iy, iz;

  /* X DIRECTION */

  for (iy = 0; iy < Ly2; iy++) {
    for (iz = 0; iz < Lz2; iz++) {
      /* backwards-going distributions e[*][0] = -1 */
      fold[Lx2-1][iy][iz][3]  = fold[1][iy][iz][3];
      fold[Lx2-1][iy][iz][8]  = fold[1][iy][iz][8];
      fold[Lx2-1][iy][iz][9]  = fold[1][iy][iz][9];
      fold[Lx2-1][iy][iz][12] = fold[1][iy][iz][12];
      fold[Lx2-1][iy][iz][13] = fold[1][iy][iz][13];
      /* forwards-going distributions e[*][0] = +1 */
      fold[0][iy][iz][1]  = fold[Lx2-2][iy][iz][1];
      fold[0][iy][iz][7]  = fold[Lx2-2][iy][iz][7];
      fold[0][iy][iz][10] = fold[Lx2-2][iy][iz][10];
      fold[0][iy][iz][11] = fold[Lx2-2][iy][iz][11];
      fold[0][iy][iz][14] = fold[Lx2-2][iy][iz][14];
    }
  }

  /* Y DIRECTION */

  for (ix = 0; ix < Lx2; ix++) {
    for (iz = 0; iz < Lz2; iz++) {
      /* backwards e[*][1] = -1 */
      fold[ix][Ly2-1][iz][4]  = fold[ix][1][iz][4];
      fold[ix][Ly2-1][iz][9]  = fold[ix][1][iz][9];
      fold[ix][Ly2-1][iz][10] = fold[ix][1][iz][10];
      fold[ix][Ly2-1][iz][13] = fold[ix][1][iz][13];
      fold[ix][Ly2-1][iz][14] = fold[ix][1][iz][14];
      /* forwards e[*][1] = +1 */
      fold[ix][0][iz][2]  = fold[ix][Ly2-2][iz][2];
      fold[ix][0][iz][7]  = fold[ix][Ly2-2][iz][7];
      fold[ix][0][iz][8]  = fold[ix][Ly2-2][iz][8];
      fold[ix][0][iz][11] = fold[ix][Ly2-2][iz][11];
      fold[ix][0][iz][12] = fold[ix][Ly2-2][iz][12];
    }
  }

  /* Z DIRECTION */

  for (ix = 0; ix < Lx2; ix++) {
    for (iy = 0; iy < Ly2; iy++) {
      /* backwards e[*][2] = -1 */
      fold[ix][iy][Lz2-1][6]  = fold[ix][iy][1][6];
      fold[ix][iy][Lz2-1][11] = fold[ix][iy][1][11];
      fold[ix][iy][Lz2-1][12] = fold[ix][iy][1][12];
      fold[ix][iy][Lz2-1][13] = fold[ix][iy][1][13];
      fold[ix][iy][Lz2-1][14] = fold[ix][iy][1][14];
      /* forwards e[*][2] = +1 */
      fold[ix][iy][0][5]  = fold[ix][iy][Lz2-2][5];
      fold[ix][iy][0][7]  = fold[ix][iy][Lz2-2][7];
      fold[ix][iy][0][8]  = fold[ix][iy][Lz2-2][8];
      fold[ix][iy][0][9]  = fold[ix][iy][Lz2-2][9];
      fold[ix][iy][0][10] = fold[ix][iy][Lz2-2][10];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  exchangeTau
 *
 *  Exchange two components in each direction.
 *
 *****************************************************************************/

void exchangeTau() {

  extern int Lx2, Ly2, Lz2;
  extern double *** tauxy;
  extern double *** tauxz;
  extern double *** tauyz;

  int ix, iy, iz;

  /* X DIRECTION: tauxy and tauxz */

  for (iy = 0; iy < Ly2; iy++) {
    for (iz = 0; iz < Lz2; iz++) {
      tauxy[Lx2-1][iy][iz] = tauxy[1][iy][iz];
      tauxz[Lx2-1][iy][iz] = tauxz[1][iy][iz];

      tauxy[0][iy][iz] = tauxy[Lx2-2][iy][iz];
      tauxz[0][iy][iz] = tauxz[Lx2-2][iy][iz];
    }
  }

  /* Y DIRECTION: tauxy and tauyz */

  for (ix = 0; ix < Lx2; ix++) {
    for (iz = 0; iz < Lz2; iz++) {
      tauxy[ix][Ly2-1][iz] = tauxy[ix][1][iz];
      tauyz[ix][Ly2-1][iz] = tauyz[ix][1][iz];

      tauxy[ix][0][iz] = tauxy[ix][Ly2-2][iz];
      tauyz[ix][0][iz] = tauyz[ix][Ly2-2][iz];
    }
  }

  /* Z DIRECTION: tauxz and tauyz */

  for (ix = 0; ix < Lx2; ix++) {
    for (iy = 0; iy < Ly2; iy++) {
      tauxz[ix][iy][Lz2-1] = tauxz[ix][iy][1];
      tauyz[ix][iy][Lz2-1] = tauyz[ix][iy][1];

      tauxz[ix][iy][0] = tauxz[ix][iy][Lz2-2];
      tauyz[ix][iy][0] = tauyz[ix][iy][Lz2-2];
    }
  }

  return;
}

#endif
