/*****************************************************************************
 *
 *  tests.h
 *
 *  Test interface.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _TESTS_H
#define _TESTS_H

void test_assert(const int);
void test_barrier(void);

#define TEST_FLOAT_TOLERANCE  1.0e-07
#define TEST_DOUBLE_TOLERANCE 1.0e-14

#endif
