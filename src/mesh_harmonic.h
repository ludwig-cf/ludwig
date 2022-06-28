//CHANGE1
/*****************************************************************************
 *
 *  mesh_harmonic.h
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

#ifndef LUDWIG_MESH_HARMONIC_H
#define LUDWIG_MESH_HARMONIC_H

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "interaction.h"

typedef struct mesh_harmonic_s mesh_harmonic_t;

int mesh_harmonic_create(pe_t * pe, cs_t * cs, mesh_harmonic_t ** pobj);
int mesh_harmonic_free(mesh_harmonic_t * obj);
int mesh_harmonic_param_set(mesh_harmonic_t * obj, double k, double r0);
int mesh_harmonic_info(mesh_harmonic_t * obj);
int mesh_harmonic_register(mesh_harmonic_t * obj, interact_t * parent);
int mesh_harmonic_compute(colloids_info_t * cinfo, void * self);
int mesh_harmonic_stats(void * self, double * stats);
int mesh_harmonic_single(mesh_harmonic_t * obj, double r, double * v, double * f);

#endif
