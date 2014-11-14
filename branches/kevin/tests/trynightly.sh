##############################################################################
#
#  nightly.sh
#
#  This is the script to run the nightly test. Nothing is done with
#  the results at the moment, except to store them.
#
#  $Id$
#
#  Edinburgh Soft Matter and Statisitcal Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#  (c) 2010 The University of Edinburgh
#
##############################################################################
#!/bin/bash

thisdir=`pwd`
testdir=trunk/tests
testscript=trytests.sh

# Log file

record=`date +%F-%T`.log

# Checkout the SVN (and send the report to the record)

svn co --username stratford http://ccpforge.cse.rl.ac.uk/svn/ludwig/trunk \
    &> $record

# Move to the test directory and run the script, sending stdout and
# stderr to record file

cd $testdir
./$testscript &> $record

# Recover the output to the present directory

cd $thisdir
mv $testdir/$record .

# Remove the evidence

rm -rf trunk
