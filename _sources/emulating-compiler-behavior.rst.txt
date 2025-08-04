Emulating Compiler Behavior
===========================

When CBI processes a file, it tries to obey all of the arguments that it can
see in the compilation database. Unfortunately, compilers often have behaviors
that are not reflected on the command line (such as their default include
paths, or compiler version macros).

If we believe (or already know!) that these behaviors will impact the
CBI's analysis of a code base, we can use a configuration file to append
additional options when emulating certain compilers.

.. attention::

    If you encounter a situation that is not supported by CBI and which cannot
    be described by our existing configuration files, please `open an issue`_.

.. _`open an issue`: https://github.com/P3HPC/code-base-investigator/issues/new/choose


Motivating Example
------------------

The ``foo.cpp`` files in our sample code base include specialization that we
have ignored so far, which selects a line based on the value of the
:code:`__GNUC__` preprocessor macro:

.. code-block:: cpp
    :linenos:
    :emphasize-lines: 6

    // Copyright (c) 2024 Intel Corporation
    // SPDX-License-Identifier: 0BSD
    #include <cstdio>

    void foo() {
    #if __GNUC__ >= 13
        printf("Using a feature that is only available in GCC 13 and later.\n");
    #else
        printf("Running the rest of foo() on the CPU.\n");
    #endif
    }

This macro is defined automatically by all GNU compilers and is set based on
the compiler's major version. For example, ``gcc`` version 13.0.0 would set
:code:`__GNUC__` to 13. Checking the values of macros like this one can be
useful when specializing code paths to workaround bugs in specific compilers,
or when specializing code paths to make use of functionality that is only
available in newer compiler versions.

Let's take another look at the compilation database entry for this file:

.. code-block:: json
    :emphasize-lines: 14

    [
    {
      "directory": "/home/username/src/build-cpu",
      "command": "/usr/bin/c++ -o CMakeFiles/tutorial.dir/main.cpp.o -c /home/username/src/main.cpp",
      "file": "/home/username/src/main.cpp"
    },
    {
      "directory": "/home/username/src/build-cpu",
      "command": "/usr/bin/c++ -o CMakeFiles/tutorial.dir/third-party/library.cpp.o -c /home/username/src/third-party/library.cpp",
      "file": "/home/username/src/third-party/library.cpp"
    },
    {
      "directory": "/home/username/src/build-cpu",
      "command": "/usr/bin/c++ -o CMakeFiles/tutorial.dir/cpu/foo.cpp.o -c /home/username/src/cpu/foo.cpp",
      "file": "/home/username/src/cpu/foo.cpp"
    }
    ]

CBI can see that the compiler used for ``foo.cpp`` is called ``/usr/bin/c++``,
but there is not enough information to decide what the value of
:code:`__GNUC__` should be.


Defining Implicit Options
-------------------------

CBI searches for a file called ``.cbi/config``, and uses the information found
in that file to determine implicit compiler options. Each compiler definition
is a TOML `table`_, of the form shown below:

.. _`table`: https://toml.io/en/v1.0.0#table

.. code:: toml

    [compiler.name]
    options = [
      "option",
      "option"
    ]

In our example, we would like to define :code:`__GNUC__` for the ``c++``
compiler, so we can add the following compiler definition:

.. code:: toml

    [compiler."c++"]
    options = [
      "-D__GNUC__=13",
    ]

.. important::
    The quotes around "c++" are necessary because of the + symbols. The quotes
    would not be necessary for other compilers.

With the :code:`__GNUC__` macro set, the two lines of code that were previously
considered "unused" are assigned to platforms, and the output of ``codebasin``
becomes:

.. code:: text

    -----------------------
    Platform Set LOC % LOC
    -----------------------
           {cpu}   8 29.63
           {gpu}   8 29.63
      {cpu, gpu}  11 40.74
    -----------------------
    Code Divergence: 0.59
    Coverage (%): 100.00
    Avg. Coverage (%): 70.37
    Total SLOC: 27


Parsing Compiler Options
------------------------

In more complex cases, emulating a compiler's implicit behavior requires CBI to
parse the command-line arguments passed to the compiler. Such emulation
requires CBI to understand which options are important and how they impact
compilation.

CBI ships with a number of compiler definitions included (see `here`_), and the
same syntax can be used to define custom compiler behaviors within the
``.cbi/config`` file.

.. _`here`: https://github.com/P3HPC/code-base-investigator/tree/main/codebasin/compilers

For example, the TOML file below defines behavior for the ``gcc`` and ``g++`` compilers:

.. code-block:: toml

    [compiler.gcc]
    # This example does not define any implicit options.

    # g++ inherits all options of gcc.
    [compiler."g++"]
    alias_of = "gcc"

    # The -fopenmp flag enables a dedicated OpenMP compiler "mode".
    [[compiler.gcc.parser]]
    flags = ["-fopenmp"]
    action = "append_const"
    dest = "modes"
    const = "openmp"

    # In OpenMP mode, the _OPENMP macro is defined.
    [[compiler.gcc.modes]]
    name = "openmp"
    defines = ["_OPENMP"]

This functionality is intended for expert users. In most cases, we expect that
defining implicit options or relying on CBI's built-in compiler emulation
support will be sufficient.

.. attention::

    If you encounter a common case where a custom compiler definition is
    required, please `open an issue`_.
