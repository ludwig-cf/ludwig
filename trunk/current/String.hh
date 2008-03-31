#ifndef STRING_HH
#define STRING_HH

//=======================================================
//
// = LIBRARY
//     Misc
//
// = FILENAME
//     String.hh
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

#include "Bool.hh"

#include <iostream>
using namespace std;

class String
//=======================================================
//
// = DESCRIPTION
//
//     The class <{String}> features basic methods to manage character
//     strings.
//
//=======================================================
{
public:

  //=======================================================
  // = CONSTRUCTOR
  //=======================================================

  String();
  // Default constructor.

  String(const String &string);
  // Copy constructor.

  String(const char *pString);
  // Constructs the string from a pointer the pointer <{pString}>.

  String(const int value);
  // Constructs the string with the integer value <{value}>.

  String(const float value);
  // Constructs the string with the float value <{value}>.

  //=======================================================
  // = DESTRUCTOR
  //=======================================================

  ~String();
  // Default destructor.

  //=======================================================
  // = OPERATORS
  //=======================================================

  void set(const String &string);
  // Sets the string from a string <{string}>.

  void set(const char *pString);
  // Sets the string from a character pointer <{pString}>.

  void set(const int value);
  // Sets the string from an integer value <{value}>.

  void set(const float value);
  // Sets the string from a float value <{value}>.

  void concat(const String &string);
  // Concats a string <{string}> to the string.

  void concat(const char *pString);
  // Concats the pointer <{pString}> to the string.

  void concat(const int value);
  // Concats the integer value <{value}> to the string.

  void concat(const float value);
  // Concats the float value <{value}> to the string.

  void concat(const char value);
  // Concats the char value <{value}> to the string.

  String& operator=(const String &string);
  // Sets the string from an other string <{string}>.

  bool operator==(const String &string) const;
  // Is the two strings equal?

  bool operator==(const char *string) const;
  // Is the two strings equal?

  String toUpperCase();
  // Builds a new string where the lower letters of the string are upper.

  String toLowerCase();
  // Builds a new string where the upper letters of the string are lower.

  //=======================================================
  // = ACCESSORS
  //=======================================================

  int length() const;
  // Returns the length of the string.

  int nbOccurences(const String &string) const;
  // Returns the number of occurences of the string <{string}>.

  char *get() const;
  // Returns the string.

  //=======================================================
  // = STREAM OPERATORS
  //=======================================================
  
  friend ostream& operator<<(ostream &os, const String &string);
  // Puts the string <{string}> onto the output stream <{os}>.

  friend istream& operator>>(istream &is, String &string);
  // Sets the string <{string}> with the current line of the file.
  //
  // <{Warning}>: if the end of the file is reached, the string
  // returned has a length equal to one.

  //=======================================================
  // = PRIVATE METHODS AND ATTRIBUTES
  //=======================================================

private:

  char *pString_;
  // Pointer into the string.

  int  len_;
  // Lenght of the string.

  const int nbDecimal_;
  // Number of decimal which has to be taken into account when
  // transforming float to string.

  char *toString(const int value);
  // Gets a string from the integer value <{value}>

  char *toString(const float value);
  // Gets a string from the float value <{value}>

};

//=======================================================
// INLINE METHODS
//=======================================================
  
//=======================================================
// = ACCESSORS
//=======================================================

inline char *String::get() const
{
  return pString_;
}

//=======================================================
inline int String::length() const
{
  return len_;
}

#endif
