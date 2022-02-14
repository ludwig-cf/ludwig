//CHANGE1
/*****************************************************************************
 *
 *  bond_harmonic2.h
 *
 *  Edinburgh Soft Matter and Statisitical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Kai Qi (kaiqi@epfl.ch)
 *
 *****************************************************************************/

#ifndef LUDWIG_BOND_HARMONIC2_H
#define LUDWIG_BOND_HARMONIC2_H

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "interaction.h"

typedef struct bond_harmonic2_s bond_harmonic2_t;

int bond_harmonic2_create(pe_t * pe, cs_t * cs, bond_harmonic2_t ** pobj);
int bond_harmonic2_free(bond_harmonic2_t * obj);
int bond_harmonic2_param_set(bond_harmonic2_t * obj, double k, double r0);
int bond_harmonic2_info(bond_harmonic2_t * obj);
int bond_harmonic2_register(bond_harmonic2_t * obj, interact_t * parent);
int bond_harmonic2_compute(colloids_info_t * cinfo, void * self);
int bond_harmonic2_stats(void * self, double * stats);
int bond_harmonic2_single(bond_harmonic2_t * obj, double r, double * v, double * f);

#endif
