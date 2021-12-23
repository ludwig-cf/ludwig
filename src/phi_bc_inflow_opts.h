/*****************************************************************************
 *
 *  phi_bc_inflow_opts.h
 *
 *  Composition boundary condition options/parameters.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_PHI_BC_INFLOW_OPTS_H
#define LUDWIG_PHI_BC_INFLOW_OPTS_H

typedef struct phi_bc_inflow_opts_s phi_bc_inflow_opts_t;

struct phi_bc_inflow_opts_s {
  double phi0;                /* Fixed value of phi at boundary */
  int flow[3];                /* Flow direction */
};

phi_bc_inflow_opts_t phi_bc_inflow_opts_default(void);

int phi_bc_inflow_opts_valid(phi_bc_inflow_opts_t options);
int phi_bc_inflow_opts_flow_valid(const int flow[3]);

#endif
