/*****************************************************************************
 *
 *  stats_colloid_force_split.c
 *
 *  Diagnostic computation of various components of the total force on
 *  a colloid. In the context of liquid crystal free energy.
 *
 *
 *  Contributing authors:
 *  Oliver Henrich (o.henrich@strath.ac.uk)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "pth_s.h"
#include "model.h"
#include "blue_phase.h"
#include "colloid_sums.h"
#include "phi_force_stress.h"
#include "stats_colloid_force_split.h"

enum {FSBULK, FSGRAD, FSCHEM}; /* Identify different components */

typedef int (*fe_lc_str_ft)(fe_lc_t * fe, int inde, double s[3][3]);

int colloid_force_from_pth(colloid_t * pc, pth_t * pth, double f[3]);
int stat_stress_compute(pth_t * pth, fe_t * fe, fe_lc_str_ft stress);
int stat_colloids_force(colloids_info_t * cinfo, pth_t * pth, int contrib);
int stat_diagnostic_write(colloid_diagnostic_t * array, int count, FILE * fp);

/*****************************************************************************
 *
 *  stats_colloid_force_split_update
 *
 *  Compute the current breakdown in stresses on the colloid.
 *  This should be called before the Beris-Edwards update in
 *  the time step, so that a consistent Q_ab is available.
 *
 *  The resulting diagnostic quantities are stored until the
 *  output stage is required (assumed to be at the end of
 *  the time step.
 *
 *****************************************************************************/

int stats_colloid_force_split_update(colloids_info_t * cinfo, fe_t * fe,
				     map_t * map,
				     field_t * q, field_grad_t * q_grad) {
  pth_t * pth = NULL;
  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe = q->pe;
  cs = q->cs;

  pth_create(pe, cs, PTH_METHOD_DIVERGENCE, &pth);

  /* Total stress */
  stat_stress_compute(pth, fe, fe_lc_stress);
  stat_colloids_force(cinfo, pth, FSCHEM);

  /* Bulk stress */
  stat_stress_compute(pth, fe, fe_lc_bulk_stress);
  stat_colloids_force(cinfo, pth, FSBULK);

  /* Gradient stress */
  stat_stress_compute(pth, fe, fe_lc_grad_stress);
  stat_colloids_force(cinfo, pth, FSGRAD);

  pth_free(pth);

  return 0;
}

/*****************************************************************************
 *
 *  stats_colloid_force_split_output
 *
 *  As we only expect this once in a while, there's a relatively simple
 *  (aka naive) mechanism for output.
 *
 *****************************************************************************/

int stats_colloid_force_split_output(colloids_info_t * cinfo, int timestep) {

  pe_t * pe = NULL;

  colloid_diagnostic_t * buf = NULL;
  MPI_Comm comm = MPI_COMM_NULL;
  int count = 0;
  int sz = sizeof(colloid_diagnostic_t);

  assert(cinfo);
  assert(timestep >= 0);

  pe = cinfo->pe;

  /* Run halo_sum on diagnostic quantities */

  colloid_sums_halo(cinfo, COLLOID_SUM_DIAGNOSTIC);

  /* Everyone count local colloids and form a send buffer. */

  colloids_info_nlocal(cinfo, &count);

  buf = (colloid_diagnostic_t *) malloc(count*sz);
  assert(buf);
  if (buf == NULL) pe_fatal(pe, "Diagnostic buffer malloc fail\n");

  {
    int nb = 0;
    colloid_t * pc = NULL;
    colloids_info_local_head(cinfo, &pc);
    for ( ; pc; pc = pc->nextlocal) {
      pc->diagnostic.index = pc->s.index; /* Record the index */
      buf[nb++] = pc->diagnostic;
    }
    assert(nb == count);
  }

  /* Communicate */
  pe_mpi_comm(pe, &comm);

  if (pe_mpi_rank(pe) > 0) {
    /* Send to root: a count, and a data buffer */
    MPI_Request req[2] = {};

    MPI_Isend(&count, 1, MPI_INT, 0, 211129, comm, req + 0);
    MPI_Isend(buf, count*sz, MPI_CHAR, 0, 211130, comm, req + 1);
    MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
    free(buf);
  }
  else {

    /* Root: output own, then those for everyone else */

    char filename[BUFSIZ] = {};
    FILE * fp = NULL;

    sprintf(filename, "colloid-diag-%8.8d.dat", timestep);
    fp = fopen(filename, "w");
    if (fp == NULL) {
      pe_fatal(pe, "Failed to open diagnostic file %s\n", filename);
    }

    stat_diagnostic_write(buf, count, fp);
    free(buf);

    for (int nr = 1; nr < pe_mpi_size(pe); nr++) {
      int rcount = 0;

      MPI_Recv(&rcount, 1, MPI_INT, nr, 211129, comm, MPI_STATUS_IGNORE);

      buf = (colloid_diagnostic_t *) malloc(rcount*sz);
      assert(buf);
      if (buf == NULL) pe_fatal(pe, "diagnostic buf malloc fail\n");

      MPI_Recv(buf, rcount*sz, MPI_CHAR, nr, 211130, comm, MPI_STATUS_IGNORE);

      /* Write out remote buffer */
      stat_diagnostic_write(buf, rcount, fp);

      free(buf);
    }

    if (ferror(fp)) perror("perror: "); /* Report but try to keep going. */
    fclose(fp);
  }

  return 0;
}

/*****************************************************************************
 *
 *  stat_diagnostic_write
 *
 *  Write out an array of count diagnostic entries to a file.
 *
 *****************************************************************************/

int stat_diagnostic_write(colloid_diagnostic_t * array, int count,
			  FILE * fp) {
  assert(array);
  assert(count >= 0);
  assert(fp);

  for (int id = 0; id < count; id++) {

    colloid_diagnostic_t * dp = array + id;

    for (int ia = 0; ia < 3; ia++) {
      /* total = hydrodynamic + non-hydrodynamic contributions */
      /*         is total force at time of update              */
      dp->ftotal[ia] = dp->fhydro[ia] + dp->fnonhy[ia];
    }
    fprintf(fp, "Colloid %6d ftotal %14.7e %14.7e %14.7e\n", dp->index,
	    dp->ftotal[X], dp->ftotal[Y], dp->ftotal[Z]);
    fprintf(fp, "Colloid %6d fhydro %14.7e %14.7e %14.7e\n", dp->index,
	    dp->fhydro[X], dp->fhydro[Y], dp->fhydro[Z]);
    fprintf(fp, "Colloid %6d fsbulk %14.7e %14.7e %14.7e\n", dp->index,
	    dp->fsbulk[X], dp->fsbulk[Y], dp->fsbulk[Z]);
    fprintf(fp, "Colloid %6d fsgrad %14.7e %14.7e %14.7e\n", dp->index,
	    dp->fsgrad[X], dp->fsgrad[Y], dp->fsgrad[Z]);
    fprintf(fp, "Colloid %6d fschem %14.7e %14.7e %14.7e\n", dp->index,
	    dp->fschem[X], dp->fschem[Y], dp->fschem[Z]);
    fprintf(fp, "Colloid %6d finter %14.7e %14.7e %14.7e\n", dp->index,
	    dp->finter[X], dp->finter[Y], dp->finter[Z]);
    fprintf(fp, "Colloid %6d fnonhy %14.7e %14.7e %14.7e\n", dp->index,
	    dp->fnonhy[X], dp->fnonhy[Y], dp->fnonhy[Z]);
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloid_force_from_pth
 *
 *****************************************************************************/

int colloid_force_from_pth(colloid_t * pc, pth_t * pth, double f[3]) {

  assert(pc);
  assert(pth);

  f[X] = 0.0; f[Y] = 0.0; f[Z] = 0.0;

  for (colloid_link_t * link = pc->lnk; link; link = link->next) {

    int id  = -1;
    int p = link->p;
    int cmod = cv[p][X]*cv[p][X] + cv[p][Y]*cv[p][Y] + cv[p][Z]*cv[p][Z];

    if (cmod != 1) continue;

    if (cv[p][X]) id = X;
    if (cv[p][Y]) id = Y;
    if (cv[p][Z]) id = Z;

    for (int ia = 0; ia < 3; ia++) {
      f[ia] += 1.0*cv[p][id]
	*pth->str[addr_rank2(pth->nsites, 3, 3, link->i, ia, id)];
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  stat_colloids_force
 *
 *****************************************************************************/

int stat_colloids_force(colloids_info_t * cinfo, pth_t * pth, int contrib) {

  colloid_t * pc = NULL;

  colloids_info_all_head(cinfo, &pc);

  for (; pc; pc = pc->nextall) {

    switch (contrib) {
    case FSCHEM:
      colloid_force_from_pth(pc, pth, pc->diagnostic.fschem);
      break;
    case FSBULK:
      colloid_force_from_pth(pc, pth, pc->diagnostic.fsbulk);
      break;
    case FSGRAD:
      colloid_force_from_pth(pc, pth, pc->diagnostic.fsgrad);
      break;
    default:
      /* Should not be here. Internal error. */
      assert(0);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  stat_stress_compute
 *
 *  Compute the stress everywhere and store using appropriate stress
 *  function.
 *
 *****************************************************************************/

int stat_stress_compute(pth_t * pth, fe_t * fe, fe_lc_str_ft stress) {

  int nlocal[3] = {};
  int nextra = 2; /* Kludge: Must be liquid crystal */

  assert(pth);
  assert(fe);

  cs_nlocal(pth->cs, nlocal);

  for (int ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (int jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (int kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	double sth[3][3] = {};

	int index = cs_index(pth->cs, ic, jc, kc);
	stress((fe_lc_t  *) fe, index, sth);
	pth_stress_set(pth, index, sth);

      }
    }
  }

  return 0;
}
