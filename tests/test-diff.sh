#!/bin/bash

###############################################################################
#
#  test-diff.sh [options] file1 file2
#
#  Examines the two log files for differences to check regression.
#  These files are assumed to be Ludwig stdout log information.
#
#  Returns 0 if files are same to required criteria, otherwise
#  the number of lines which differ.
#  Negative return value indicates problem with arguments.
#
#  We have to get rid of some stuff that will not match:
#    - the run times
#    - SVN version information
#    - exact location of input file
#    - allow "Model R" tests to pass by looking for "d3q19 R" etc
#
#  Options:
#    -v causes the actual results of the diff to be sent to stdout
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  Contributing Authors:
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#  (c) 2013-2019 The University of Edinburgh
#
###############################################################################

# Slightly flaky. We assume location of tests to find the floating
# point 'diff' script

FPDIFF=../../awk-fp-diff.sh
TESTDIFF=test-diff.sh

# Check input

if [ $# -lt 2 ]
then
    echo "Usage: $0 referece-file test-file"
    exit -1
fi

is_verbose=0

while getopts v opt
do
case "$opt" in
    v) is_verbose=1;;
esac
done

if [ $is_verbose -eq 1 ]; then
    shift
fi

if [ ! -e $1 ]; then
    if [ $is_verbose -eq 1 ]; then
	echo "$TESTDIFF: $1: No such file or directory"
    fi
    exit -1
fi

if [ ! -e $2 ]; then
    if [ $is_verbose -eq 1 ]; then
	echo "$TESTDIFF: $2: No such file or directory"
    fi
    exit -1
fi

# Get rid of:
#   - line with the versiosn number "Welcome to Ludwig"
#   - timer statistics identified via "call)" or "calls)"
#   - blank lines
#   - "Timer resolution"
#   - exact location of the input file via "user parameters"  

sed '/call)/d' $1 > test-diff-tmp.ref
sed -i~ '/calls)/d' test-diff-tmp.ref
sed -i~ '/Welcome/d' test-diff-tmp.ref
sed -i~ '/Target thread model: None/d' test-diff-tmp.ref
sed -i~ '/^$/d' test-diff-tmp.ref
sed -i~ '/Timer/d' test-diff-tmp.ref
sed -i~ '/user.parameters.from/d' test-diff-tmp.ref
sed -i~ 's/d2q9\ R/d2q9/' test-diff-tmp.ref
sed -i~ 's/d3q15\ R/d3q15/' test-diff-tmp.ref
sed -i~ 's/d3q19\ R/d3q19/' test-diff-tmp.ref
sed -i~ '/GPU\ INFO/d' test-diff-tmp.ref
sed -i~ '/SIMD\ vector/d' test-diff-tmp.ref

sed '/call)/d' $2 > test-diff-tmp.log
sed -i~ '/calls)/d' test-diff-tmp.log
sed -i~ '/Welcome/d' test-diff-tmp.log
sed -i~ '/SVN.revision/d' test-diff-tmp.log
sed -i~ '/^$/d' test-diff-tmp.log
sed -i~ '/Timer/d' test-diff-tmp.log
sed -i~ '/user.parameters.from/d' test-diff-tmp.log
sed -i~ 's/d2q9\ R/d2q9/' test-diff-tmp.log
sed -i~ 's/d3q15\ R/d3q15/' test-diff-tmp.log
sed -i~ 's/d3q19\ R/d3q19/' test-diff-tmp.log
sed -i~ '/GPU\ INFO/d' test-diff-tmp.log
sed -i~ '/SIMD\ vector/d' test-diff-tmp.log

# Here we use the floating point diff to measure "success"

var=`$FPDIFF test-diff-tmp.ref test-diff-tmp.log | wc -l`


if [ $is_verbose -eq 1 -a $var -gt 0 ]
    then
    c=`$FPDIFF test-diff-tmp.ref test-diff-tmp.log`
    echo "$c"
fi

rm -rf test-diff-tmp.ref test-diff-tmp.log

exit $var

