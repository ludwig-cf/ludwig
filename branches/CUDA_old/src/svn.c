/****************************************************************************
 *
 *  svn.c
 *
 *  $Id$
 *
 *  If the preprocessor identifier SVN_REVISION is defined, it must
 *  be a quoted string, as it is intended to hold the result of the
 *  utility `svnversion`. E.g., compile with (shudder)
 *
 *   -D SVN_REVISION='"'`svnversion`'"'
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 ****************************************************************************/

#include "svn.h"

#if defined (SVN_REVISION)
static const char * svn_revision_ = SVN_REVISION;
#else
static const char * svn_revision_ = "unknown";
#endif 

/****************************************************************************
 *
 *  svn_revision
 *
 *  Return a description of the SVN revision as a string.
 *
 ****************************************************************************/

const char * svn_revision(void) {

  return svn_revision_;
}

