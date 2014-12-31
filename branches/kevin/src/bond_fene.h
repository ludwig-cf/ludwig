/*****************************************************************************
 *
 *  bond_fene.h
 *
 *  Edinburgh Soft Matter and Statisitical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2014)
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef BOND_FENE_H
#define BOND_FENE_H

#include "coords.h"
#include "colloids.h"
#include "interaction.h"

typedef struct bond_fene_s bond_fene_t;

int bond_fene_create(coords_t * cs, bond_fene_t ** pobj);
int bond_fene_free(bond_fene_t * obj);
int bond_fene_param_set(bond_fene_t * obj, double k, double r0);
int bond_fene_info(bond_fene_t * obj);
int bond_fene_register(bond_fene_t * obj, interact_t * parent);
int bond_fene_compute(colloids_info_t * cinfo, void * self);
int bond_fene_stats(void * self, double * stats);
int bond_fene_single(bond_fene_t * obj, double r, double * v, double * f);

#endif
