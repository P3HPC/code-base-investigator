Sample Code Base
================

This tutorial uses a sample code base designed to showcase the features of CBI.

.. attention::

   To follow along with the tutorial, we first need a copy of the sample code
   base.

   It can be downloaded from :download:`here<sample-code-base.zip>` or copied
   from the ``docs/source/sample-code-base`` directory `on GitHub`_.

.. _on GitHub: https://github.com/P3HPC/code-base-investigator/tree/main/docs/sample-code-base/


Directory Structure
-------------------

The sample code base consists of just a few source files, arranged as shown
below::

    src/
    ├── CMakeLists.txt
    ├── cpu
    │   └── foo.cpp
    ├── gpu
    │   └── foo.cpp
    ├── main.cpp
    └── third-party
        ├── library.cpp
        └── library.h


Although simple, this structure is representative of many applications that
target multiple platforms. The code base contains:

- A directory (``src``) containing all of the source files required to build
  the application.

- Two subdirectories (``cpu`` and ``gpu``) containing source files that are
  only used when building the application to target specific platforms.

- Some shared source files (``main.cpp``) that are always used when building
  the application to target any platform.

- Some third-party source files (``third-party/library.h`` and
  ``third-party/library.cpp``).

.. tip::
    Generally speaking, "third party source files" just means "source files
    maintained by somebody else". Even if we're working with a code base
    without any external dependencies, treating code written by other
    developers or other teams as "third party source files" will allow us to
    limit our analysis to source files that we care about.


File Structure
--------------

Let's take a look at one of the files in the code base, ``main.cpp``:

.. code-block:: cpp
   :linenos:
   :emphasize-lines: 10,12

    // Copyright (c) 2024 Intel Corporation
    // SPDX-License-Identifier: 0BSD
    #include <cstdio>
    #include "third-party/library.h"
    void foo();

    int main(int argc, char* argv[])
    {
    #if !defined(GPU_OFFLOAD)
        printf("Running on the CPU.\n");
    #else
        printf("Running on the GPU.\n");
    #endif
        foo();
        bar();
    }

The preprocessor directives on Lines 9, 11 and 12 (:code:`#if`, :code:`#else`
and :code:`#endif`) define a specialization point, allowing Lines 10 and 12
to be specialized based on the value of a preprocessor macro
(:code:`GPU_OFFLOAD`). This approach is common in many C, C++ and Fortran
applications.

Lines 3, 4 and 5 also define potential specialization points, because the
compilation commands targeting different platforms may search different
include paths and/or link against different libraries.

.. note::

    Although this file contains 16 lines of *text*, CBI will count only 13
    lines of *code*. Comments and whitespace do not count.
