#ifndef BOOL_HH
#define BOOL_HH

//=======================================================
//
// = LIBRARY
//     Misc
//
// = FILENAME
//     Bool.hh
//
// = AUTHOR(S)
//     Alexandre Dupuis
//
// = VERSION
//     $Revision: 1.1.1.1 $
//
// = DATE RELEASED
//     $Date: 2008-03-31 08:18:40 $
//
// = COPYRIGHT
//     University of Geneva, Switzerland
//
//=======================================================

// ------------------------------------------------------------
// Compiling with SUN CC
// ------------------------------------------------------------

// SUN CC defines `false' and `true', but not type `bool'

#ifdef __SUNPRO_CC

enum bool {false = 0, true = 1};

#endif

// ------------------------------------------------------------
// Compiling with g++
// ------------------------------------------------------------
#ifdef __GNUC__

// Check if g++ >= 2.6. (Recent version have builtin bool, older
// versions don't have it.)
#if !(__GNUC__ > 2 || __GNUC_MINOR__ >= 6)
enum bool {false = 0, true = 1};
#endif

#endif

// ------------------------------------------------------------
// Compiling with mpCC under the IBM SP2
// ------------------------------------------------------------

#ifdef SP

enum bool {false = 0, true = 1};

#endif

// ------------------------------------------------------------
// Place next compiler features here...
// ------------------------------------------------------------

#endif
