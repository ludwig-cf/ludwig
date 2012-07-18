//=======================================================
//
// = LIBRARY
//     Misc
//
// = FILENAME
//     String.cc
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

#include "String.hh"
#include <cstring>

//=======================================================
// = CONSTRUCTOR
//=======================================================

String::String() : nbDecimal_(3)
{ 
  len_=0;
  pString_=0;
  set("");
}

//=======================================================
String::String(const String &string) : nbDecimal_(3)
{
  len_=0;
  pString_=0;
  set(string);
}

//=======================================================

String::String(const char *pString) : nbDecimal_(3)
{ 
  len_=0;
  pString_=0;
  set(pString);
}

//=======================================================

String::String(const int value) : nbDecimal_(3)
{ 
  len_=0;
  pString_=0;
  set(toString(value));
}

//=======================================================

String::String(const float value) : nbDecimal_(3)
{ 
  len_=0;
  pString_=0;
  set(toString(value));
}

//=======================================================
// = DESTRUCTOR
//=======================================================

String::~String() 
{ 
  if (pString_ != 0) delete pString_;
  pString_=0;
}

//=======================================================
// = OPERATORS
//=======================================================
  
void String::set(const String &string)
{
  set(string.get());
}

//=======================================================

void String::set(const char *pString)
{
  if (pString_ != 0) { 
    delete pString_; 
    pString_=0;
  }
  len_=strlen(pString);
  len_++;
  pString_=new char[len_];
  strcpy(pString_,pString);
  pString_[len_-1]=(char) 0;
}

//=======================================================

void String::set(const int value)
{
  set(toString(value));
}

//=======================================================

void String::set(const float value)
{
  set(toString(value));
}

//=======================================================

void String::concat(const String &string)
{
  concat(string.get());
}
  
//=======================================================

void String::concat(const char *pString)
{
  char *tmpString;

  // Computes the new length
  len_+=strlen(pString);
  //len_++;

  // Allocates the new string
  tmpString=new char[len_+1];

  // Copies the two parts of the new string
  strcpy(tmpString,pString_);
  strcat(tmpString,pString);
  tmpString[len_]=(char) 0;

  // Switchs the string
  delete [] pString_;
  pString_=tmpString;
}

//=======================================================

void String::concat(const int value)
{
  concat(toString(value));
}

//=======================================================

void String::concat(const float value)
{
  concat(toString(value));
}

//=======================================================

void String::concat(const char value)
{
  char string[2];
  string[0]=value;
  string[1]=(char) 0;

  concat(string);
}

//=======================================================

String& String::operator=(const String &string)
{
  set(string);

  return *this;
}

//=======================================================
String String::toLowerCase()
{
  String result;

  int i;
  char c;
  for (i=0;i<len_;i++) {
    c=pString_[i];
    if ((c >= 65) && (c <= 90)) c+=32;
    result.concat(c);
  }

  return result;
}

//=======================================================
String String::toUpperCase()
{
  String result;

  int i;
  char c;
  for (i=0;i<len_;i++) {
    c=pString_[i];
    if ((c >= 97) && (c <= 122)) c-=32;
    result.concat(c);
  }

  return result;
}

//=======================================================
// = ACCESSORS
//=======================================================

int String::nbOccurences(const String &string) const
{
  int result=0;
  char *s=string.get();
  int len=strlen(s);

  int i,k;
  for (i=0;i<len_;i++) {
    if (pString_[i] == s[0]) {
      k=0;
      while(k < len) {
        if (s[k] != pString_[i+k]) { k=len+1; break; }
        k++;
      }
      if (k == len) result++;
    }
  }

  return result;
}

//=======================================================
  
bool String::operator==(const String &string) const
{
  if (len_ != string.length()) return false;

  char *stringToCompare;
  stringToCompare=new char[len_];
  strcpy(stringToCompare,string.get());

  int i=0;
  while (i < len_) {
    if (pString_[i] != stringToCompare[i]) return false;
    i++;
  }

  return true;
}

//=======================================================
bool String::operator==(const char *string) const
{
  return *this==String(string);
}

//=======================================================
// = STREAM OPERATORS
//=======================================================

ostream& operator<<(ostream &os, const String &string)
{
  return os << string.get();
}

//=======================================================
istream& operator>>(istream &is, String &string)
{
  string.set("");

  if (is.eof()) return is;

  char c=0;

  is.get(c);
  while ((c != 10) && (is.eof() == 0)){
    string.concat(c);
    is.get(c);
  }

  return is;
}

//=======================================================
// = PRIVATE OPERATORS
//=======================================================

char *String::toString(const int value)
{
  int div=1;
  int len=1;
  int isNegative=0;
  char *string=0;
  int localValue;

  // Determines the sign
  if (value < 0) {
    localValue=-value;
    isNegative=1;
  }
  else localValue=value;

  // Finds the length of the string
  while ((int) (localValue/div) >= 10) { div*=10; len++; }

  // Allocates the string
  string=new char[len+1+isNegative];

  // If the value is negative
  if (isNegative == 1) string[0]='-';

  int number=0;
  int mul=0;
  int i;
  for (i=0;i<len;i++) {

    // Computes the current number
    number=(int) (localValue/div)-mul*10;
    div=div/10;
    mul=mul*10+number;
  
    // Stores the number at the right place
    string[i+isNegative]=(char) 48+number;
  }

  // Indicates the end of the string
  string[len+isNegative]=(char) 0;

  // Returns the result
  return string;
}
  
//=======================================================
char *String::toString(const float value)
{
  int isNegative=0;
  char *string=0;
  float localValue;
  int exhibitor=0;
  int isExhibitorNegative=0;

  // Determines the sign
  if (value < 0) {
    localValue=-value;
    isNegative=1;
  }
  else localValue=value;

  // Determines the exhibitor
  if (localValue < 1.0) {
    while(localValue < 1.0) {
      localValue*=10.0;
      exhibitor--;
    }
    isExhibitorNegative=1;
  }
  else
    while(localValue >= 10.0) {
      localValue/=10.0;
      exhibitor++;
    }

  // Converts the exhibitor into string
  char *exhibitorString;
  exhibitorString=toString(exhibitor);

  // Finds the length of the string
  int len=0;
  len=isNegative+2+nbDecimal_+4+strlen(exhibitorString);

  // Allocates the string
  string=new char[len];
  string[0]=(char) 0;

  // Converts the decimal part into string
  char *decimalPartString;
  float multiplicator=1.0;
  int m=0;
  while (m < nbDecimal_) { m++; multiplicator*=10.0; }
  decimalPartString=toString((int) 
			     ((localValue-(int) localValue)*multiplicator));

  // Constructs the string
  char firstDigit[2];
  firstDigit[0]=(char) 48+((int) localValue);
  firstDigit[1]=(char) 0;
  if (isNegative == 1) strcat(string,"-");
  strcat(string,firstDigit);
  strcat(string,".");
  strcat(string,decimalPartString);
  strcat(string," E");
  if (isExhibitorNegative == 0) strcat(string,"+");
  strcat(string,exhibitorString);

  return string;
}

