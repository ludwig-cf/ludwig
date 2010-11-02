/*****************************************************************************
 *
 *  tests.h
 *
 *  $Id: tests.h,v 1.2 2010-11-02 17:51:22 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef TESTS_H
#define TESTS_H

void test_assert(const int lvalue);

#define TEST_FLOAT_TOLERANCE  1.0e-07
#define TEST_DOUBLE_TOLERANCE 1.0e-14

#endif
