/*****************************************************************************
 *
 *  model_le.c
 *
 *  Lees-Edwards transformations for distributions.
 *
 *  Note that the distributions have displacement u*t
 *  not u*(t-1) returned by le_get_displacement().
 *  This is for reasons of backwards compatability.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  J.-C. Desplat and Ronojoy Adhikari developed the reprojection method.
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "control.h"
#include "coords.h"
#include "model_le.h"
#include "pe.h"
#include "physics.h"
#include "timer.h"
#include "util.h"

#include "leesedwards.h"
#include "target.h"

__global__ static void le_reproject(lb_t *lb, lees_edw_t *le, kernel_ctxt_t * ktxt);
static void le_displace_and_interpolate(lb_t *lb, lees_edw_t *le);
__global__ static void interpolation(lb_t *lb, lees_edw_t *le, int *positive,  
    int *negative, int nprop, int negprop, int displacement, double t, kernel_ctxt_t * ktxt);
__global__ static void copy_back(lb_t *lb, lees_edw_t *le, int *positive,  
    int *negative, int nprop, int negprop, int displacement, kernel_ctxt_t * ktxt);
static int le_displace_and_interpolate_parallel(lb_t *lb, lees_edw_t *le);

void copyModelToDevice(lb_model_t *h_model, lb_model_t *d_model) {
    int nvel = h_model->nvel;
    // Allocate memory on the GPU for the arrays in the struct
    int8_t (*d_cv)[3];
    double *d_wv;
    double *d_na;

    tdpMalloc((void**)&d_cv, sizeof(int8_t[3]) * nvel);
    tdpMalloc((void**)&d_wv, sizeof(double) * nvel);
    tdpMalloc((void**)&d_na, sizeof(double) * nvel);
    
    // Copy the data from host to the GPU
    tdpMemcpy(d_cv, h_model->cv, sizeof(int8_t[3]) * nvel, tdpMemcpyHostToDevice);
    tdpMemcpy(d_wv, h_model->wv, sizeof(double) * nvel, tdpMemcpyHostToDevice);
    tdpMemcpy(d_na, h_model->na, sizeof(double) * nvel, tdpMemcpyHostToDevice);
    
    // Set the pointers in the struct to the newly allocated GPU memory
    tdpMemcpy(&(d_model->cv), &d_cv, sizeof(int8_t(*)[3]), tdpMemcpyHostToDevice);
    tdpMemcpy(&(d_model->wv), &d_wv, sizeof(double*), tdpMemcpyHostToDevice);
    tdpMemcpy(&(d_model->na), &d_na, sizeof(double*), tdpMemcpyHostToDevice);

    //copy the rest data to gpu
    tdpMemcpy(&(d_model->ndim), &(h_model->ndim), sizeof(int8_t), tdpMemcpyHostToDevice);
    tdpMemcpy(&(d_model->nvel), &(h_model->nvel), sizeof(int8_t), tdpMemcpyHostToDevice);
    tdpMemcpy(&(d_model->cs2), &(h_model->cs2), sizeof(double), tdpMemcpyHostToDevice);
}

cudaError_t copy_buffer_duy_to_device(lees_edw_s* d_lees_edw, int* h_buffer_duy, size_t nxbuffer) {
    // First, allocate memory on the device for the buffer_duy array
    int* d_buffer_duy;
    cudaError_t err = tdpMalloc((void**) &d_buffer_duy, nxbuffer * sizeof(int));

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for buffer_duy (error code %s)!\n", cudaGetErrorString(err));
        return err;
    }

    // Then, copy the data from the host array to the newly allocated device array
    err = tdpMemcpy(d_buffer_duy, h_buffer_duy, nxbuffer * sizeof(int), tdpMemcpyHostToDevice);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy buffer_duy from host to device (error code %s)!\n", cudaGetErrorString(err));
        return err;
    }

    // Finally, update the pointer in the device structure to point to the new device array
    err = tdpMemcpy(&(d_lees_edw->buffer_duy), &d_buffer_duy, sizeof(int*), tdpMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy buffer_duy pointer to device structure (error code %s)!\n", cudaGetErrorString(err));
        return err;
    }

    return cudaSuccess;
}

__global__ void interpolation(lb_t *lb, lees_edw_t *le, int *positive,  
    int *negative, int nprop, int negprop, int displacement, double t, kernel_ctxt_t * ktxt) {
    int plane, ic, jc, kc;
    int nhalo, ndist, nplane;
    int jdy, j1, j2, index0, index1;
    double dy, fr;
    double ltot[3];
    int nlocal[3];

    lb_ndist(lb, &ndist);
    lees_edw_nlocal(le, nlocal);
    lees_edw_nhalo(le, &nhalo);
    lees_edw_ltot(le, ltot);
    nplane = lees_edw_nplane_local(le);

    int kindex;
    int kiter;
    assert(ktxt);
    kiter = kernel_iterations(ktxt);

    for_simt_parallel(kindex, kiter, 1) {
        plane = kernel_coords_ic(ktxt, kindex);
        jc = kernel_coords_jc(ktxt, kindex);
        kc = kernel_coords_kc(ktxt, kindex);

        if (jc <= nlocal[Y] && kc <= nlocal[Z] && plane < nplane) {

            ic = lees_edw_plane_location(le, plane);
            lees_edw_buffer_displacement(le, nhalo, t, &dy);
            dy = fmod(dy, ltot[Y]);
            jdy = floor(dy);
            fr = dy - jdy;

            j1 = 1 + (jc + jdy - 1 + 2 * nlocal[Y]) % nlocal[Y];
            j2 = 1 + (j1 % nlocal[Y]);

            index0 = lees_edw_index(le, ic, j1, kc);
            index1 = lees_edw_index(le, ic, j2, kc);

            /* xdisp_fwd_cv[0] identifies cv[p][X] = +1 */

            for (int n = 0; n < ndist; n++) {
                for (int i = 0; i < nprop; i++) {
                    int l0 = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index0, n, positive[i]);
                    int l1 = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index1, n, positive[i]);
                    int index = ((jc-1)*nlocal[Z] + (kc-1))*ndist*nprop + n*nprop + i + 2 * plane * displacement;
                    lb->recv_buff[index] = (1.0 - fr) * lb->f[l0] + fr * lb->f[l1];
                }
            }

            /* OTHER DIRECTION */
            ic = lees_edw_plane_location(le, plane) + 1;
            lees_edw_buffer_displacement(le, nhalo, t, &dy);
            dy = fmod(-dy, ltot[Y]);
            jdy = floor(dy);
            fr = dy - jdy;

            j1 = 1 + (jc + jdy - 1 + 2 * nlocal[Y]) % nlocal[Y];
            j2 = 1 + (j1 % nlocal[Y]);

            index0 = lees_edw_index(le, ic, j1, kc);
            index1 = lees_edw_index(le, ic, j2, kc);

            /* xdisp_fwd_cv[0] identifies cv[p][X] = -1 */

            for (int n = 0; n < ndist; n++) {
                for (int i = 0; i < negprop; i++) {
                    int l0 = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index0, n, negative[i]);
                    int l1 = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index1, n, negative[i]);
                    int index = ((jc-1)*nlocal[Z] + (kc-1))*ndist*negprop + n*negprop + i + (2 * plane + 1) * displacement;
                    lb->recv_buff[index] = (1.0 - fr) * lb->f[l0] + fr * lb->f[l1];
                }
            }
        }
    }
}

__global__ void copy_back(lb_t *lb, lees_edw_t *le, int *positive,  
    int *negative, int nprop, int negprop, int displacement, kernel_ctxt_t * ktxt) {
    int plane, ic, jc, kc;
    int nhalo, ndist, nplane;
    int nlocal[3];
    int index0;

    lb_ndist(lb, &ndist);
    lees_edw_nlocal(le, nlocal);
    lees_edw_nhalo(le, &nhalo);
    nplane = lees_edw_nplane_local(le);

    int kindex;
    int kiter;
    assert(ktxt);
    kiter = kernel_iterations(ktxt);

    for_simt_parallel(kindex, kiter, 1) {
        plane = kernel_coords_ic(ktxt, kindex);
        jc = kernel_coords_jc(ktxt, kindex);
        kc = kernel_coords_kc(ktxt, kindex);

        if (jc <= nlocal[Y] && kc <= nlocal[Z] && plane < nplane) {
            ic = lees_edw_plane_location(le, plane);
            index0 = lees_edw_index(le, ic, jc, kc);

            for (int n = 0; n < ndist; n++) {
                for (int i = 0; i < nprop; i++) {
                    int index = ((jc-1)*nlocal[Z] + (kc-1))*ndist*nprop + n*nprop + i + 2 * plane * displacement;
                    int la = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index0, n, positive[i]);
                    lb->f[la] = lb->recv_buff[index];
                }
            }

            /* Another direction */

            ic = lees_edw_plane_location(le, plane) + 1; 
            index0 = lees_edw_index(le, ic, jc, kc);

            for (int n = 0; n < ndist; n++) {
                for (int i = 0; i < negprop; i++) {
                    int index = ((jc-1)*nlocal[Z] + (kc-1))*ndist*negprop + n*negprop + i + (2 * plane + 1) * displacement;
                    int la = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index0, n, negative[i]);
                    lb->f[la] = lb->recv_buff[index];
                }
            }
        }
    }  
}

/*****************************************************************************
 *
 *  lb_le_apply_boundary_conditions
 *
 *  This is the driver to apply the LE conditions to the distributions
 *  (applied to the post-collision distributions). There are two
 *  stages:
 *
 *  1. a reprojection of distributions that will cross a plane in the
 *     upcoming propagation step.
 *  2. a displacement and interpolation of the reprojected distributions
 *     to take account of the sliding displacement as a function of time.
 *
 *  Note we never deal with the halo regions here, as we assume the
 *  upcoming propagation will be immediately preceeded by a distribution
 *  halo update.
 *
 *****************************************************************************/

__host__ int lb_le_apply_boundary_conditions(lb_t *lb, lees_edw_t *le) {
    int mpi_cartsz[3];

    assert(lb);
    assert(le);

    lees_edw_cartsz(le, mpi_cartsz);

    if (lees_edw_nplane_local(le) > 0) {
        TIMER_start(TIMER_LE);

        /* Everything must be done on host at the moment (slowly) ... */
        /* ... and copy back at the end */
        lees_edw_t * le_target;
        int ndevice;
        tdpGetDeviceCount(&ndevice);
        
        if (ndevice > 0) {
            copyModelToDevice(&lb->model, &lb->target->model);
            
            lees_edw_target(le, &le_target);
            copy_buffer_duy_to_device(le_target, le->buffer_duy, le->param->nxbuffer);
        }
       
        

        int nlocal[3], nplane;
        lees_edw_nlocal(le, nlocal);
        nplane = lees_edw_nplane_local(le);

        //tdp
        dim3 nblk, ntpb;
        kernel_info_t limits;
        kernel_ctxt_t * ctxt = NULL;
        limits.imin = 0; limits.imax = 2 * nplane - 1;
        limits.jmin = 1; limits.jmax = nlocal[Y];
        limits.kmin = 1; limits.kmax = nlocal[Z];

        kernel_ctxt_create(le->cs, NSIMDVL, limits, &ctxt);
        kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

        tdpLaunchKernel(le_reproject, nblk, ntpb, 0, 0, lb->target, le_target, ctxt->target);
        tdpAssert(tdpPeekAtLastError());
        tdpAssert(tdpDeviceSynchronize());

        if (mpi_cartsz[Y] > 1) {
            le_displace_and_interpolate_parallel(lb, le);
        }
        else {
            le_displace_and_interpolate(lb, le);
        }
        
        TIMER_stop(TIMER_LE);
    }

    return 0;
}

/*****************************************************************************
 *
 *  le_reproject
 *
 *  This is the reprojection of the post collision distributions to
 *  take account of the velocity jump at the planes.
 *
 *  We compute the moments, and then the change to the moments:
 *
 *     rho  -> rho (unchanged)
 *     g_a  -> g_a +/- rho u^le_a
 *     S_ab -> S_ab +/- rho u_a u^le_b +/- rho u_b u^le_a + rho u^le_a u^le_b
 *
 *  with analogous expressions for order parameter moments.
 *
 *  The change to the distribution is then computed by a reprojection.
 *  Ghost modes are unchanged.
 *
 *****************************************************************************/

__global__ static void le_reproject(lb_t *lb, lees_edw_t *le, kernel_ctxt_t * ktxt) {
    int ic, jc, kc, side, index;
    int nplane, plane;
    int ia, ib;
    int nlocal[3];
    int n, ndist;
    int8_t cx = 0;
    double rho, ds[3][3], udotc, sdotq;
    double g[3], du[3];
    double fnew;
    double t;
    physics_t *phys = NULL;

    assert(lb);
    assert(le);
    assert(ktxt);

    lb_ndist(lb, &ndist);
    nplane = lees_edw_nplane_local(le);
    physics_ref(&phys);
    lees_edw_nlocal(le, nlocal);
    t = 1.0 * physics_control_timestep(phys);
    
    int kindex;
    int kiter;
    kiter = kernel_iterations(ktxt);

    for_simt_parallel(kindex, kiter, 1) {
        side = kernel_coords_ic(ktxt, kindex);
        jc = kernel_coords_jc(ktxt, kindex);
        kc = kernel_coords_kc(ktxt, kindex);

        if (jc <= nlocal[Y] && kc <= nlocal[Z] && side < 2 * nplane) {
            du[X] = 0.0;
            du[Y] = 0.0;
            du[Z] = 0.0;

            plane = side / 2;
            if (side % 2 == 0) {
                /* Start with plane below Lees-Edwards BC */
                lees_edw_plane_uy_now(le, t, &du[Y]);
                du[Y] *= -1.0;
                ic = lees_edw_plane_location(le, plane);
                cx = +1;
            }
            else {
                /* Finally, deal with plane above LEBC */
                lees_edw_plane_uy_now(le, t, &du[Y]);
                ic = lees_edw_plane_location(le, plane) + 1;
                cx = -1;
            }

            index = lees_edw_index(le, ic, jc, kc);

            for (n = 0; n < ndist; n++) {

                /* Compute 0th and 1st moments */
                lb_dist_enum_t ndn = (lb_dist_enum_t)n;
                lb_0th_moment(lb, index, ndn, &rho);
                lb_1st_moment(lb, index, ndn, g);

                for (ia = 0; ia < 3; ia++) {
                    for (ib = 0; ib < 3; ib++) {
                        ds[ia][ib] = (g[ia] * du[ib] + du[ia] * g[ib] + rho * du[ia] * du[ib]);
                    }
                }

                /* Now update the distribution */
                for (int p = 1; p < lb->model.nvel; p++) {

                    double cs2 = lb->model.cs2;
                    double rcs2 = 1.0 / cs2;
                    if (lb->model.cv[p][X] != cx)
                        continue;

                    udotc = du[Y] * lb->model.cv[p][Y];
                    sdotq = 0.0;

                    for (ia = 0; ia < 3; ia++) {
                        for (ib = 0; ib < 3; ib++) {
                            double dab = cs2 * (ia == ib);
                            double q = (lb->model.cv[p][ia] * lb->model.cv[p][ib] - dab);
                            sdotq += ds[ia][ib] * q;
                        }
                    }

                    /* Project all this back to the distribution. */

                    lb_f(lb, index, p, n, &fnew);
                    fnew += lb->model.wv[p] * (rho * udotc * rcs2 + 0.5 * sdotq * rcs2 * rcs2);
                    lb_f_set(lb, index, p, n, fnew);
                }
            }
        }
            /* next site */
    }
    return;
}

/*****************************************************************************
 *
 *  le_displace_and_interpolate
 *
 *  For each side of each plane, work out the relevant displacement
 *  and do the necessary interpolation to get the modified plane-
 *  crossing distributions.
 *
 *****************************************************************************/

void le_displace_and_interpolate(lb_t *lb, lees_edw_t *le) {

    int nlocal[3];
    int nplane;
    int ndist;
    int nprop, negprop;
    int displacement;
    double t;
    physics_t *phys = NULL;
    lees_edw_t * le_target;

    assert(lb);
    assert(le);

    lees_edw_nlocal(le, nlocal);
    nplane = lees_edw_nplane_local(le);
    physics_ref(&phys);
    lb_ndist(lb, &ndist);
    lees_edw_target(le, &le_target);

    t = 1.0 * physics_control_timestep(phys);

    /* We need to interpolate into a temporary buffer to make sure we
     * don't overwrite distributions taking part. The size is just
     * determined by the size of the local domain, and the number
     * of plane-crossing distributions. */

    /* Allocate a buffer large enough for all cvp[][X] = +1 */
    nprop = 0;
    negprop = 0;
    for (int p = 1; p < lb->model.nvel; p++) {
        if (lb->model.cv[p][X] == +1) nprop += 1;
        if (lb->model.cv[p][X] == -1) negprop += 1;
    }
    displacement = ndist * nprop * nlocal[Y] * nlocal[Z];

    // record the displacement of propgation
    int *positive = (int *)malloc(sizeof(int) * nprop);
    int *negative = (int *)malloc(sizeof(int) * negprop);
    for (int p = 1, i = 0, j = 0; p < lb->model.nvel; p++) {
        if (lb->model.cv[p][X] == +1) {
            positive[i] = p;
            i++;
        }
        if (lb->model.cv[p][X] == -1) {
            negative[j] = p;
            j++;
        }
    }

    // copy the displacement array to the device
    int *d_positive, *d_negative;
    tdpMalloc((void**)&d_positive, sizeof(int) * nprop);
    tdpMalloc((void**)&d_negative, sizeof(int) * negprop);
    tdpMemcpy(d_positive, positive, sizeof(int) * nprop, tdpMemcpyHostToDevice);
    tdpMemcpy(d_negative, negative, sizeof(int) * negprop, tdpMemcpyHostToDevice);

    //tdp
    dim3 nblk, ntpb;
    kernel_info_t limits;
    kernel_ctxt_t * ctxt = NULL;
    limits.imin = 0; limits.imax = nplane - 1;
    limits.jmin = 1; limits.jmax = nlocal[Y];
    limits.kmin = 1; limits.kmax = nlocal[Z];

    kernel_ctxt_create(le->cs, NSIMDVL, limits, &ctxt);
    kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

    tdpLaunchKernel(interpolation, nblk, ntpb, 0, 0, lb->target, le_target, 
        d_positive, d_negative, nprop, negprop, displacement, t, ctxt->target);
    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());

    tdpLaunchKernel(copy_back, nblk, ntpb, 0, 0, lb->target, le_target, 
        d_positive, d_negative, nprop, negprop, displacement, ctxt->target);
    tdpAssert(tdpPeekAtLastError());
    tdpAssert(tdpDeviceSynchronize());

    kernel_ctxt_free(ctxt);

    free(positive);
    free(negative);
    tdpFree(d_positive);
    tdpFree(d_negative);

    return;
}

/*****************************************************************************
 *
 *  le_displace_and_interpolate_parallel
 *
 *  Here we need to communicate to be able to do the displacement of
 *  the buffers in the along-plane (Y-) direction.
 *
 *  Locally, we need to find interpolated values of the plane-crossing
 *  distributions for 1 <= jc <= nlocal[Y]. To do a linear interpolation
 *  everywhere, this requires (nlocal[Y] + 1) points displaced in the
 *  appropriate direction.
 *
 *  Likewise, we need to send a total of (nlocal[Y] + 1) points to the
 *  two corresponding recieving processes. Note we never involve the
 *  halo regions here (so a preceeding halo exchange is not required).
 *
 *****************************************************************************/

static int le_displace_and_interpolate_parallel(lb_t *lb, lees_edw_t *le) {

    int ic, jc, kc;
    int j1, j1mod;
    int jdy;
    int n1, n2;
    int ndata, ndata1, ndata2;
    int nhalo;
    int ind0, ind1, ind2, index;
    int n, nplane, plane;
    int ntotal[3];
    int nlocal[3];
    int offset[3];
    int nrank_s[3], nrank_r[3];
    int nprop;
    int ndist;

    const int tag1 = 3102;
    const int tag2 = 3103;

    double fr;
    double dy;
    double t;
    double ltot[3];
    double *send_buff;
    double *recv_buff;

    physics_t *phys = NULL;
    MPI_Comm comm;
    MPI_Request req[4];
    MPI_Status status[4];

    assert(lb);
    assert(le);

    lees_edw_ltot(le, ltot);
    lees_edw_ntotal(le, ntotal);
    lees_edw_nlocal(le, nlocal);
    lees_edw_nhalo(le, &nhalo);
    lees_edw_nlocal_offset(le, offset);

    nplane = lees_edw_nplane_local(le);
    lees_edw_comm(le, &comm);

    physics_ref(&phys);

    t = 1.0 * physics_control_timestep(phys);
    lb_ndist(lb, &ndist);

    nprop = 0;
    for (int p = 1; p < lb->model.nvel; p++) {
        if (lb->model.cv[p][X] == +1)
            nprop += 1;
    }

    ndata = ndist * nprop * nlocal[Y] * nlocal[Z];
    send_buff = (double *)malloc(ndata * sizeof(double));
    assert(send_buff);
    if (send_buff == NULL)
        pe_fatal(lb->pe, "malloc(send_buff) failed\n");

    ndata = ndist * nprop * (nlocal[Y] + 1) * nlocal[Z];
    recv_buff = (double *)malloc(ndata * sizeof(double));
    assert(recv_buff);
    if (recv_buff == NULL)
        pe_fatal(lb->pe, "malloc(recv_buff) failed\n");

    for (plane = 0; plane < nplane; plane++) {

        ic = lees_edw_plane_location(le, plane);

        lees_edw_buffer_displacement(le, nhalo, t, &dy);
        dy = fmod(dy, ltot[Y]);
        jdy = floor(dy);
        fr = dy - jdy;

        /* Starting y coordinate is j1: 1 <= j1 <= ntotal[y] */

        jc = offset[Y] + 1;
        j1 = 1 + (jc + jdy - 1 + 2 * ntotal[Y]) % ntotal[Y];
        lees_edw_jstart_to_mpi_ranks(le, j1, nrank_s, nrank_r);

        j1mod = 1 + (j1 - 1) % nlocal[Y];
        n1 = (nlocal[Y] - j1mod + 1);
        n2 = j1mod;

        ndata1 = n1 * nlocal[Z] * ndist * nprop;
        ndata2 = n2 * nlocal[Z] * ndist * nprop;

        /* Post the receives */

        MPI_Irecv(recv_buff, ndata1, MPI_DOUBLE, nrank_r[0], tag1, comm, req);
        MPI_Irecv(recv_buff + ndata1, ndata2, MPI_DOUBLE, nrank_r[1], tag2, comm, req + 1);

        /* Load the send buffer. Note that data at j1mod gets sent to both
         * receivers, making up the total of (nlocal[Y] + 1) points */

        ndata = 0;
        for (jc = 1; jc <= nlocal[Y]; jc++) {
            for (kc = 1; kc <= nlocal[Z]; kc++) {

                /* cv[p][X] = +1 identified by disp_fwd[] */
                index = lees_edw_index(le, ic, jc, kc);

                for (n = 0; n < ndist; n++) {
                    for (int p = 1; p < lb->model.nvel; p++) {
                        if (lb->model.cv[p][X] != +1)
                            continue;
                        int ijkp = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index, n, p);
                        send_buff[ndata++] = lb->f[ijkp];
                    }
                }
                /* Next site */
            }
        }

        ndata = ndata2 - nlocal[Z] * ndist * nprop;

        MPI_Issend(send_buff + ndata, ndata1, MPI_DOUBLE, nrank_s[0], tag1, comm, req + 2);
        MPI_Issend(send_buff, ndata2, MPI_DOUBLE, nrank_s[1], tag2, comm, req + 3);

        /* Wait for the receives, and sort out the interpolated values */

        MPI_Waitall(2, req, status);

        for (jc = 1; jc <= nlocal[Y]; jc++) {
            for (kc = 1; kc <= nlocal[Z]; kc++) {

                index = lees_edw_index(le, ic, jc, kc);
                ind0 = ndist * nprop * ((jc - 1) * nlocal[Z] + (kc - 1));

                for (n = 0; n < ndist; n++) {
                    ind1 = ind0 + n * nprop;
                    ind2 = ind0 + ndist * nprop * nlocal[Z] + n * nprop;
                    for (int p = 1; p < lb->model.nvel; p++) {
                        if (lb->model.cv[p][X] != +1)
                            continue;
                        int ijk = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index, n, p);
                        lb->f[ijk] = (1.0 - fr) * recv_buff[ind1++] + fr * recv_buff[ind2++];
                    }
                }
                /* Next site */
            }
        }

        /* Finish the sends */
        MPI_Waitall(2, req + 2, status);

        /* NOW THE OTHER DIRECTION */

        ic = lees_edw_plane_location(le, plane) + 1;

        lees_edw_buffer_displacement(le, nhalo, t, &dy);
        dy = fmod(-dy, ltot[Y]);
        jdy = floor(dy);
        fr = dy - jdy;

        /* Starting y coordinate (global address): range 1 <= j1 <= ntotal[Y] */

        jc = offset[Y] + 1;
        j1 = 1 + (jc + jdy - 1 + 2 * ntotal[Y]) % ntotal[Y];
        lees_edw_jstart_to_mpi_ranks(le, j1, nrank_s, nrank_r);

        j1mod = 1 + (j1 - 1) % nlocal[Y];
        n1 = (nlocal[Y] - j1mod + 1);
        n2 = j1mod;

        ndata1 = n1 * nlocal[Z] * ndist * nprop;
        ndata2 = n2 * nlocal[Z] * ndist * nprop;

        /* Post the receives */

        MPI_Irecv(recv_buff, ndata1, MPI_DOUBLE, nrank_r[0], tag1, comm, req);
        MPI_Irecv(recv_buff + ndata1, ndata2, MPI_DOUBLE, nrank_r[1], tag2, comm, req + 1);

        /* Load the send buffer. Note that data at j1mod gets sent to both
         * receivers, making up the total of (nlocal[Y] + 1) points */

        ndata = 0;
        for (jc = 1; jc <= nlocal[Y]; jc++) {
            for (kc = 1; kc <= nlocal[Z]; kc++) {

                /* cv[p][X] = -1 identified by disp_bwd[] */
                index = lees_edw_index(le, ic, jc, kc);

                for (n = 0; n < ndist; n++) {
                    for (int p = 1; p < lb->model.nvel; p++) {
                        if (lb->model.cv[p][X] != -1)
                            continue;
                        int ijkp = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index, n, p);
                        send_buff[ndata++] = lb->f[ijkp];
                    }
                }
                /* Next site */
            }
        }

        ndata = ndata2 - nlocal[Z] * ndist * nprop;

        MPI_Issend(send_buff + ndata, ndata1, MPI_DOUBLE, nrank_s[0], tag1, comm, req + 2);
        MPI_Issend(send_buff, ndata2, MPI_DOUBLE, nrank_s[1], tag2, comm, req + 3);

        /* Wait for the receives, and interpolate from the buffer */

        MPI_Waitall(2, req, status);

        for (jc = 1; jc <= nlocal[Y]; jc++) {
            for (kc = 1; kc <= nlocal[Z]; kc++) {

                index = lees_edw_index(le, ic, jc, kc);
                ind0 = ndist * nprop * ((jc - 1) * nlocal[Z] + (kc - 1));

                for (n = 0; n < ndist; n++) {
                    ind1 = ind0 + n * nprop;
                    ind2 = ind0 + ndist * nprop * nlocal[Z] + n * nprop;
                    for (int p = 1; p < lb->model.nvel; p++) {
                        if (lb->model.cv[p][X] != -1)
                            continue;
                        int ijk = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index, n, p);
                        lb->f[ijk] = (1.0 - fr) * recv_buff[ind1++] + fr * recv_buff[ind2++];
                    }
                }
                /* Next site */
            }
        }

        /* Mop up the sends */
        MPI_Waitall(2, req + 2, status);
    }

    free(send_buff);
    free(recv_buff);

    return 0;
}

/*****************************************************************************
 *
 *  model_le_init_shear_profile
 *
 *  Initialise the distributions to be consistent with a steady-state
 *  linear shear profile, consistent with plane velocity.
 *
 *****************************************************************************/

int lb_le_init_shear_profile(lb_t *lb, lees_edw_t *le) {

    int ic, jc, kc, index;
    int i, j, p;
    int nlocal[3];
    double rho0, u[NDIM], gradu[NDIM][NDIM];
    double eta;

    physics_t *phys = NULL;

    assert(lb);
    assert(le);

    pe_info(lb->pe, "Initialising shear profile\n");

    /* Initialise the density, velocity, gradu; ghost modes are zero */

    physics_ref(&phys);
    physics_rho0(phys, &rho0);
    physics_eta_shear(phys, &eta);

    lees_edw_nlocal(le, nlocal);

    for (i = 0; i < lb->model.ndim; i++) {
        u[i] = 0.0;
        for (j = 0; j < lb->model.ndim; j++) {
            gradu[i][j] = 0.0;
        }
    }

    lees_edw_shear_rate(le, &gradu[X][Y]);

    /* Loop trough the sites */

    for (ic = 1; ic <= nlocal[X]; ic++) {

        lees_edw_steady_uy(le, ic, &u[Y]);

        /* We can now project the physical quantities to the distribution */

        for (jc = 1; jc <= nlocal[Y]; jc++) {
            for (kc = 1; kc <= nlocal[Z]; kc++) {

                index = lees_edw_index(le, ic, jc, kc);

                for (p = 0; p < lb->model.nvel; p++) {
                    double f = 0.0;
                    double cdotu = 0.0;
                    double sdotq = 0.0;
                    double cs2 = lb->model.cs2;
                    double rcs2 = 1.0 / cs2;

                    for (i = 0; i < lb->model.ndim; i++) {
                        cdotu += lb->model.cv[p][i] * u[i];
                        for (j = 0; j < lb->model.ndim; j++) {
                            double dij = (i == j);
                            double qij = lb->model.cv[p][i] * lb->model.cv[p][j] - cs2 * dij;
                            sdotq += (rho0 * u[i] * u[j] - eta * gradu[i][j]) * qij;
                        }
                    }
                    f = lb->model.wv[p] * (rho0 + rcs2 * rho0 * cdotu + 0.5 * rcs2 * rcs2 * sdotq);
                    lb_f_set(lb, index, p, 0, f);
                }
                /* Next site */
            }
        }
    }

    return 0;
}
