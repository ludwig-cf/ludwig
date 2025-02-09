/*****************************************************************************
 *
 *  test_colloid_link.c
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>

#include "pe.h"
#include "colloid_link.h"

int test_colloid_link_max_2d_d2q9(void);
int test_colloid_link_max_3d_d3q15(void);
int test_colloid_link_max_3d_d3q19(void);
int test_colloid_link_max_3d_d3q27(void);

/*****************************************************************************
 *
 *  test_colloid_link_suite
 *
 *****************************************************************************/

int test_colloid_link_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_colloid_link_max_2d_d2q9();
  test_colloid_link_max_3d_d3q15();
  test_colloid_link_max_3d_d3q19();
  test_colloid_link_max_3d_d3q27();

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_colloid_link_max_2d_d2q9
 *
 *  Maximum number of links for a disk with radius a in the d2q9 model.
 *  We consider disks with centre at three different symmetry points of
 *  the lattice:
 *
 *       .   .   .      .   .   .     .   .   .
 *                                          Z
 *       .   X   .      .   . Y .     .   .   .
 *
 *       .   .   .      .   .   .     .   .   .
 *
 *  with the origin being at a lattice point (X). The lattice vector
 *  joining the colloid centre to a lattice site determines a set of
 *  radii at which new lattice sites are incorporated into the
 *  discrete surface.
 *
 *  X case: centre at (0,0) The numbers are by drawing a diagram.
 *
 *  Vector      Radius       No. inside      Links
 *  ----------------------------------------------
 *              a > 0                 1          8
 *  (1, 0)      a > sqrt(  1)         5         24
 *  (1, 1)      a > sqrt(  2)         9         32
 *  (2, 0)      s > sqrt(  4)        13         40
 *  (2, 1)      a > sqrt(  5)        21         48
 *  (2, 2)      a > sqrt(  8)        25         56
 *  (3, 0)      a > sqrt(  9)        29         64
 *  (3, 1)      a > sqrt( 10)        37         64
 *  (3, 2)      a > sqrt( 13)        45         72
 *  (4, 0)      a > sqrt( 16)        49         80
 *  (4, 1)      a > sqrt( 17)        57         80
 *  (3, 3)      a > sqrt( 18)        61         88
 *  (4, 2)      a > sqrt( 20)        69         88
 *  (4, 3)      a > sqrt( 25)
 *  (5, 0)      a > sqrt( 25)        81        104
 *  ...
 *  (5, 5)      a > sqrt( 50)       161        144
 *  (9, 9)      a > sqrt(162)       509        248
 *  (15,15)     a > sqrt(450)      1425        416
 *
 *  Y case: centre at (1/2, 0)
 *
 *  Vector      Radius       No. inside      Links
 *  ----------------------------------------------
 *  (1/2, 0)    a > sqrt( 1/4)        2         14
 *  (1/2, 1)    a > sqrt( 5/4)        6         26
 *  (3/2, 0)    a > sqrt( 9/4)        8         30
 *  (3/2, 1)    a > sqrt(13/4)       12         38
 *  (1/2, 2)    a > sqrt(17/4)       16         42
 *  (3/2, 2)    a > sqrt(25/4)        -                             -
 *  (5/2, 0)    a > sqrt(25/4)       22         54
 *
 *  Z case: centre at (1/2, 1/2)
 *
 *  Vector      Radius       No. inside      Links
 *  ----------------------------------------------
 *  (1/2, 1/2)  a > sqrt( 2/4)        4         20
 *
 *
 *  It is enough to check the X case.
 *
 *****************************************************************************/

int test_colloid_link_max_2d_d2q9(void) {

  int ifail = 0;
  int nvel  = 9;

  /* X case */
  {
    int nlink = 24;
    double a = sqrt(1.0);
    int nmax = colloid_link_max_2d(a, nvel);
    if (nmax < nlink) ifail = 1;
    assert(ifail == 0);
  }

  {
    int nlink = 64;
    double a = sqrt(10.0);

    int nmax = colloid_link_max_2d(a, nvel);
    if (nmax < nlink) ifail = -1;
    assert(ifail == 0);
  }

  /* The more interesting cases are a > sqrt(16) */
  {
    int nlink = 248;
    double a = sqrt(162.0);

    int nmax = colloid_link_max_2d(a, nvel);
    if (nmax < nlink) ifail = -1;
    assert(ifail == 0);
  }

  {
    int nlink = 416;
    double a = sqrt(450.0);

    int nmax = colloid_link_max_2d(a, nvel);
    if (nmax < nlink) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_links_max_d3q15
 *
 *  Drawing a diagram is less practical than in 2d. There are many
 *  possibilities.
 *
 *  X case: centre at (0,0,0)
 *
 *  Vector        Radius       No. inside      Links
 *  ------------------------------------------------
 *                a > 0                 1         14
 *  (1, 0, 0)     a > sqrt(  1)         7         86
 *  (1, 1, 0)     a > sqrt(  2)        19        158
 *  (1, 1, 1)     a > sqrt(  3)        27        206
 *  (2, 0, 0)     a > sqrt(  4)        33        230
 *  (2, 1, 0)     a > sqrt(  5)        57        374
 *  (2, 1, 1)     a > sqrt(  6)        81        422
 *  (2, 2, 0)     a > sqrt(  8)        93        494
 *  (2, 2, 1)     a > sqrt(  9)       123        614
 *  (3, 1, 0)     a > sqrt( 10)       147        662
 *  (3, 1, 1)     a > sqrt( 11)       171        710
 *  (2, 2, 2)     a > sqrt( 12)       179        710
 *  (3, 2, 1)     a > sqrt( 14)       251        950
 *  (4, 0, 0)     a > sqrt( 16)       257        974
 *  ...
 *  (4, 4, 4)     a > sqrt( 48)      1365       2894
 *  (9, 9, 9)     a > sqrt(243)     15895      15230
 *
 *****************************************************************************/

int test_colloid_link_max_3d_d3q15(void) {

  int ifail = 0;
  int nvel = 15;

  {
    int nlink = 86;
    double a = sqrt(1.0);

    int nmax = colloid_link_max_3d(a, nvel);
    if (nmax < nlink) ifail = -1;
    assert(ifail == 0);
  }

  {
    int nlink = 158;
    double a = sqrt(2.0);

    int nmax = colloid_link_max_3d(a, nvel);
    if (nmax < nlink) ifail = -1;
    assert(ifail == 0);
  }

  /* Larger radii ... */
  {
    int nlink = 2894;
    double a = sqrt(48.0);

    int nmax = colloid_link_max_3d(a, nvel);
    if (nmax < nlink) ifail = -1;
    assert(ifail == 0);
  }

  {
    int nlink = 15230;
    double a = sqrt(243.0);

    int nmax = colloid_link_max_3d(a, nvel);
    if (nmax < nlink) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_links_max_d3q19
 *
 *  The lattice vectors are the same as d3q15.
 *
 *  X case: centre at (0,0,0)
 *
 *  Vector        Radius          a0     inside      Links
 *  ------------------------------------------------------
 *                a > 0                       1         18
 *  (1, 0, 0)     a > sqrt(  1)   1.001       7         90
 *  (1, 1, 0)     a > sqrt(  2)   1.415      19        186
 *  ...
 *  (4, 4, 4)     a > sqrt( 48)   6.930    1365       3354
 *  (9, 9, 9)     a > sqrt(243)  15.590   15895      17562
 *
 *****************************************************************************/

int test_colloid_link_max_3d_d3q19(void) {

  int ifail = 0;
  int nvel = 19;

  {
    int nlink = 86;
    double a = sqrt(1.0);

    int nmax = colloid_link_max_3d(a, nvel);
    if (nmax < nlink) ifail = -1;
    assert(ifail == 0);
  }

  {
    int nlink = 158;
    double a = sqrt(2.0);

    int nmax = colloid_link_max_3d(a, nvel);
    if (nmax < nlink) ifail = -1;
    assert(ifail == 0);
  }

  /* Larger radii ... */
  {
    int nlink = 3354;
    double a = sqrt(48.0);

    int nmax = colloid_link_max_3d(a, nvel);
    if (nmax < nlink) ifail = -1;
    assert(ifail == 0);
  }

  {
    int nlink = 17562;
    double a = sqrt(243.0);

    int nmax = colloid_link_max_3d(a, nvel);
    if (nmax < nlink) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_links_max_d3q27
 *
 *  The lattice vectors are the same as d3q15.
 *
 *  X case: centre at (0,0,0)
 *
 *  Vector        Radius          a0     inside      Links
 *  ------------------------------------------------------
 *                a > 0                       1         18
 *  (1, 0, 0)     a > sqrt(  1)   1.001       7        146
 *  (1, 1, 0)     a > sqrt(  2)   1.415      19        290
 *  ...
 *  (4, 4, 4)     a > sqrt( 48)   6.930    1365       5378
 *  (9, 9, 9)     a > sqrt(243)  15.590   15895      28226
 *
 *****************************************************************************/

int test_colloid_link_max_3d_d3q27(void) {

  int ifail = 0;
  int nvel = 27;

  {
    int nlink = 146;
    double a = sqrt(1.0);

    int nmax = colloid_link_max_3d(a, nvel);
    if (nmax < nlink) ifail = -1;
    assert(ifail == 0);
  }

  {
    int nlink = 290;
    double a = sqrt(2.0);

    int nmax = colloid_link_max_3d(a, nvel);
    if (nmax < nlink) ifail = -1;
    assert(ifail == 0);
  }

  /* Larger radii ... */
  {
    int nlink = 5378;
    double a = sqrt(48.0);

    int nmax = colloid_link_max_3d(a, nvel);
    if (nmax < nlink) ifail = -1;
    assert(ifail == 0);
  }

  {
    int nlink = 28226;
    double a = sqrt(243.0);

    int nmax = colloid_link_max_3d(a, nvel);
    if (nmax < nlink) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}
