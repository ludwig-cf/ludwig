##############################################################################
#
#  repository-dump.sh
#
#  This script backs up the SVN repository via svnadmin dump.
#  As we are remote, this requires first getting a local copy of the
#  whole repository, done using svnsync.
#
#  Could be, e.g., invoked by cron.
#
#  $Id$
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#  (c) 2011 The University of Edinburgh
#
##############################################################################
#!/bin/bash

copydirectory=/home/kevin/nightly/copy
dumpfile=/home/kevin/nightly/repository-`date +%F`.dmp

# Make a temporary local repository in $copydirectory

svnadmin create $copydirectory

# Allow svn to make revision changes to $copydirectory

echo '#!/bin/sh' > $copydirectory/hooks/pre-revprop-change
chmod +x $copydirectory/hooks/pre-revprop-change

# Sync the new copy with existing repository

svnsync init file://$copydirectory http://ccpforge.cse.rl.ac.uk/svn/ludwig
svnsync sync file://$copydirectory

# Dump the whole thing to a file and remove the copy

svnadmin dump $copydirectory > $dumpfile
rm -rf $copydirectory

