Understanding Specialization
============================

The goal of CBI is to help developers to reason about how a code base
uses *specialization* to adapt to the capabilities and requirements of the
different platforms it supports. By measuring specialization, we can reason
about its impact upon maintenance effort.


Platforms
#########

The definition of *platform* used by CBI was first introduced in
"`Implications of a Metric for Performance Portability`_", and is shared
with the `P3 Analysis Library`_:

  A collection of software and hardware on which an *application* may run a
  *problem*.

.. _Implications of a Metric for Performance Portability:
   https://doi.org/10.1016/j.future.2017.08.007

.. _P3 Analysis Library:
   https://intel.github.io/p3-analysis-library/

This definition is deliberately very flexible, so a platform can represent
**any** execution environment for which code may be specialized. A platform
could be a compiler, an operating system, a micro-architecture or some
combination of these options.


Specialization
##############

There are many forms of specialization. What they all have in common is that
these *specialization points* act as branches: different code is executed
on different platforms based on some set of conditions. These conditions
express a platform's capabilities, properties of the input problem, or both.

The simplest form of specialization point is a run-time branch, which is easily
expressed but can incur run-time overheads and prevent compiler optimizations.
Compile-time specialization avoids these issues, and in practice a lot of
specialization is performed using preprocessor tools or with some kind of
metaprogramming.


Code Divergence
###############

Code divergence is a metric proposed by Harrell and Kitson in "`Effective
Performance Portability`_", which uses the Jaccard distance to measure the
distance between two source codes.

For a given set of platforms, :math:`H`, the code divergence :math:`CD` of
an application :math:`a` solving problem :math:`p` is an average of
pairwise distances:

.. math::
    CD(a, p, H) = \binom{|H|}{2}^{-1}
                  \sum_{\{i, j\} \in \binom{H}{2}} {d_{i, j}(a, p)}

where :math:`d_{i, j}(a, p)` represents the distance between the source
code required by platforms :math:`i` and :math:`j` for application
:math:`a` to solve problem :math:`p`.

The distance is calculated as:

.. math::
    d_{i, j}(a, p) = 1 - \frac{|c_i(a, p) \cap c_j(a, p)|}
                              {|c_i(a, p) \cup c_j(a, p)|}

where :math:`c_i` and :math:`c_j` are the lines of code required to compile
application :math:`a` and solve problem :math:`p` using platforms :math:`i`
and :math:`j`. A distance of 0 means that all code is shared between the
two platforms, whereas a distance of 1 means that no code is shared.

.. note::

    It is sometimes useful to talk about code *convergence* instead, which is
    simply the code divergence subtracted from 1.

.. _Effective Performance Portability:
    https://doi.org/10.1109/P3HPC.2018.00006
