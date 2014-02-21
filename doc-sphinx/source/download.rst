.. _download:

*********
Download
*********
Current release: CHOMPACK 1.1.1 (October 23, 2011)

==================
Source
==================

CHOMPACK is implemented in C with a 
Python/`CVXOPT <http://abel.ee.ucla.edu/cvxopt>`_ interface. 
It can be downloaded as a `zip file <chompack-1.1.1.zip>`_
containing the source files and installation guidelines. 

=====================
Binary distributions
=====================

-------------
Linux
-------------

A 64-bit binary distribution for Linux is available `here`__.
To unpack and install the binary distribution for all users, issue the following command:

__ chompack-1.1.1.linux-x86_64.tar.gz

::
  
    $ sudo tar -xvf chompack-1.1.1.linux-x86_64.tar.gz -C /

This will install CHOMPACK in :file:`/usr/local/lib/python2.7/dist-packages/`.

-------------
Mac OS X
-------------


64-bit binary distributions for Mac OS X Lion are avialiable with

   - non-optimized reference BLAS/LAPACK (`tar.gz`__)
   - ATLAS optimized for an Intel Core i5 1.7 GHz CPU (`tar.gz`__)

__  chompack-1.1.1.macosx-10.7-x86_64.tar.gz 
__  chompack-1.1.1.macosx-10.7-atlas-x86_64.tar.gz 

To unpack and install the binary distribution for all users, issue the following command:

::
  
    $ sudo tar -xvf chompack-1.1.1.macosx-10.7-x86_64.tar.gz -C /

This will install CHOMPACK in :file:`/Library/Python/2.7/site-packages/`.
