#!/bin/bash

###############################################################################
#
#  test-dif.sh [options] file1 file2
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
#
#  Options:
#    -v causes the actual results of the diff to be sent to stdout
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  Contributing Authors:
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#  (c) 2013 The University of Edinburgh
#
###############################################################################


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

if [ ! -e $1 -o ! -e $2 ]; then
    echo "File arguments incorrect: $1 $2"
    exit -1
fi

# Get rid of:
#   - timer statistics identified via "call)" or "calls)"
#   - SVN revision information identified via "SVN revision"
#   - exact location of the input file via "user parameters"

sed '/call)/d' $1 > test-diff-tmp.ref
sed -i '/calls)/d' test-diff-tmp.ref
sed -i '/SVN.revision/d' test-diff-tmp.ref
sed -i '/user.parameters.from/d' test-diff-tmp.ref

sed '/call)/d' $2 > test-diff-tmp.log
sed -i '/calls)/d' test-diff-tmp.log
sed -i '/SVN.revision/d' test-diff-tmp.log
sed -i '/user.parameters.from/d' test-diff-tmp.log

var=`diff test-diff-tmp.ref test-diff-tmp.log | wc -l`

if [ $is_verbose -eq 1 ]
    then
    c=`diff test-diff-tmp.ref test-diff-tmp.log`
    echo "$c"
fi

rm -rf test-diff-tmp.ref test-diff-tmp.log

exit $var

