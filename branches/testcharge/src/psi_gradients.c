/****************************************************************************
 *
 *  psi_gradients.c
 *
 *  Finite difference stencils used in the 
 *  electrokinetic routines.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *  (c) 2014 The University of Edinburgh
 *
 ****************************************************************************/

#include "psi_gradients.h"

#ifdef NP_D3Q6
const int psi_gr_cv[PSI_NGRAD][3] = {{ 0,  0,  0},
		{-1,  0,  0}, { 1,  0,  0}, { 0, -1,  0},
		{ 0,  1,  0}, { 0,  0, -1}, { 0,  0,  1}};

#define w0 (0.0)
#define w1  (1.0/6.0)

const double psi_gr_wv[PSI_NGRAD] = {w0,
                         	     w1, w1, w1, 
			             w1, w1, w1};

const double psi_gr_rnorm[PSI_NGRAD] = {0.0,
		                        1.0, 1.0, 1.0,
                	                1.0, 1.0, 1.0};

const double psi_gr_rcs2 = 3.0;
#endif

#ifdef NP_D3Q18
const int psi_gr_cv[PSI_NGRAD][3] = {{ 0,  0,  0},
		{ 1,  1,  0}, { 1,  0,  1}, { 1,  0,  0},
		{ 1,  0, -1}, { 1, -1,  0}, { 0,  1,  1},
		{ 0,  1,  0}, { 0,  1, -1}, { 0,  0,  1},
		{ 0,  0, -1}, { 0, -1,  1}, { 0, -1,  0},
		{ 0, -1, -1}, {-1,  1,  0}, {-1,  0,  1},
		{-1,  0,  0}, {-1,  0, -1}, {-1, -1,  0}};


#define w0 (12.0/36.0)
#define w1  (2.0/36.0)
#define w2  (1.0/36.0)

#define sqrt2 1.4142135623730951 

const double psi_gr_wv[PSI_NGRAD] = {w0,
                        w2, w2, w1, w2, w2, w2, 
			w1, w2, w1, w1, w2, w1, 
			w2, w2, w2, w1, w2, w2};

const double psi_gr_rnorm[PSI_NGRAD] = {0.0,
		1.0/sqrt2, 1.0/sqrt2, 1.0, 1.0/sqrt2, 1.0/sqrt2, 1.0/sqrt2,
		1.0, 1.0/sqrt2, 1.0, 1.0, 1.0/sqrt2, 1.0, 
		1.0/sqrt2, 1.0/sqrt2, 1.0/sqrt2, 1.0, 1.0/sqrt2, 1.0/sqrt2};

const double psi_gr_rcs2 = 3.0;
#endif

#ifdef NP_D3Q26
const int psi_gr_cv[PSI_NGRAD][3] = {{ 0, 0, 0}, 
		{-1,-1,-1}, {-1,-1, 0}, {-1,-1, 1}, 
		{-1, 0,-1}, {-1, 0, 0}, {-1, 0, 1}, 
		{-1, 1,-1}, {-1, 1, 0}, {-1, 1, 1}, 
		{ 0,-1,-1}, { 0,-1, 0}, { 0,-1, 1}, 
		{ 0, 0,-1},             { 0, 0, 1}, 
		{ 0, 1,-1}, { 0, 1, 0}, { 0, 1, 1}, 
		{ 1,-1,-1}, { 1,-1, 0}, { 1,-1, 1}, 
		{ 1, 0,-1}, { 1, 0, 0}, { 1, 0, 1}, 
		{ 1, 1,-1}, { 1, 1, 0}, { 1, 1, 1}};

#define w0 (8.0/27.0)
#define w1 (2.0/27.0)
#define w2 (1.0/54.0)
#define w3 (1.0/216.0)

#define sqrt2 1.4142135623730951 
#define sqrt3 1.7320508075688772

const double psi_gr_wv[PSI_NGRAD] = {w0,
		w3, w2, w3, 
		w2, w1, w2, 
		w3, w2, w3, 
		w2, w1, w2, 
		w1,     w1, 
		w2, w1, w2, 
		w3, w2, w3, 
		w2, w1, w2, 
		w3, w2, w3};     

const double psi_gr_rnorm[PSI_NGRAD] = {0.0,
		1.0/sqrt3, 1.0/sqrt2, 1.0/sqrt3, 
		1.0/sqrt2, 1.0, 1.0/sqrt2, 
		1.0/sqrt3, 1.0/sqrt2, 1.0/sqrt3, 
		1.0/sqrt2, 1.0, 1.0/sqrt2, 
		1.0,                  1.0, 
		1.0/sqrt2, 1.0, 1.0/sqrt2, 
		1.0/sqrt3, 1.0/sqrt2, 1.0/sqrt3, 
		1.0/sqrt2, 1.0, 1.0/sqrt2, 
		1.0/sqrt3, 1.0/sqrt2, 1.0/sqrt3};   

const double psi_gr_rcs2 = 3.0;
#endif
