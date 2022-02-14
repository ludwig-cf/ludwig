//CHANGE1
/*****************************************************************************
 *
 *  bond_harmonic3.h
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

#ifndef LUDWIG_BOND_HARMONIC3_H
#define LUDWIG_BOND_HARMONIC3_H

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "interaction.h"

typedef struct bond_harmonic3_s bond_harmonic3_t;

int bond_harmonic3_create(pe_t * pe, cs_t * cs, bond_harmonic3_t ** pobj);
int bond_harmonic3_free(bond_harmonic3_t * obj);
int bond_harmonic3_param_set(bond_harmonic3_t * obj, double k, double r0);
int bond_harmonic3_info(bond_harmonic3_t * obj);
int bond_harmonic3_register(bond_harmonic3_t * obj, interact_t * parent);
int bond_harmonic3_compute(colloids_info_t * cinfo, void * self);
int bond_harmonic3_stats(void * self, double * stats);
int bond_harmonic3_single(bond_harmonic3_t * obj, double r, double * v, double * f);

#endif
