# Copyright (C) 2019-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Contains functions for generating command-line reports.
"""

import filecmp
import hashlib
import itertools as it
import logging
import numbers
import os
import re
import string
import sys
import warnings
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Self, TextIO

from tabulate import tabulate

from codebasin import CodeBase, util
from codebasin.finder import ParserState
from codebasin.preprocessor import CodeNode

log = logging.getLogger(__name__)


def _heading(text: str, stream: TextIO):
    """
    Parameters
    ----------
    text: str
        The text to use as the heading.

    stream: TextIO
        The stream the heading will eventually be written to.

    Returns
    -------
    str
        A heading string appropriately formatted for the output stream.
    """
    if stream.isatty():
        return f"\033[1m\033[4m{text}\033[0m\n"
    else:
        underline = "=" * len(text)
        return f"{text}\n{underline}"


def extract_platforms(setmap):
    """
    Extract a list of unique platforms from a set map
    """
    unique_platforms = set(it.chain.from_iterable(setmap.keys()))
    return list(unique_platforms)


def coverage(
    setmap: dict[frozenset[str], int],
    platforms: set[str] | None = None,
) -> float:
    """
    Compute the percentage of lines in `setmap` required by at least one
    platform in the supplied `platforms` set.

    Parameters
    ----------
    setmap: dict[frozenset[str], int]
        A mapping from platform set to SLOC count.

    platforms: set[str], optional
        The set of platforms to use when computing coverage.
        If not provided, computes coverage for all platforms.

    Returns
    -------
    float
        The amount of code used by at least one platform, as a percentage.
        If `setmap` contains no lines of code or no platforms, returns NaN.
    """
    if not platforms:
        platforms = set().union(*setmap.keys())

    used = 0
    total = 0
    for subset, sloc in setmap.items():
        total += sloc
        if subset == frozenset():
            continue
        elif any([p in platforms for p in subset]):
            used += sloc

    if total == 0:
        return float("nan")

    return (used / total) * 100.0


def average_coverage(
    setmap: dict[frozenset[str], int],
    platforms: set[str] | None = None,
) -> float:
    """
    Computes the coverage for each platform in the supplied `platforms` set
    (by calling :py:func:`coverage` for each platform), then returns the
    average (mean) of these values.

    Parameters
    ----------
    setmap: dict[frozenset[str], int]
        A mapping from platform set to SLOC count.

    platforms: set[str], optional
        The set of platforms to use when computing coverage.
        If not provided, computes average over all platforms.

    Returns
    -------
    float
        The average amount of code used by each platform, as a percentage.
        If `setmap` contains no lines of code or no platforms, returns NaN.
    """
    if not platforms:
        platforms = set().union(*setmap.keys())

    if len(platforms) == 0:
        return float("nan")

    total = sum([coverage(setmap, {p}) for p in platforms])
    return total / len(platforms)


def distance(setmap, p1, p2) -> float:
    """
    Compute distance between two platforms
    """
    total = 0
    for pset, count in setmap.items():
        if (p1 in pset) or (p2 in pset):
            total += count
    d = 0
    for pset, count in setmap.items():
        if (p1 in pset) ^ (p2 in pset):
            d += count / float(total)
    return d


def divergence(setmap) -> float:
    """
    Compute code divergence as defined by Harrell and Kitson
    i.e. average of pair-wise distances between platform sets
    """
    platforms = extract_platforms(setmap)

    d = 0.0
    npairs = 0
    for p1, p2 in it.combinations(platforms, 2):
        d += distance(setmap, p1, p2)
        npairs += 1

    if npairs == 0:
        return float("nan")
    return d / float(npairs)


def summary(setmap: dict[frozenset[str], int], stream: TextIO = sys.stdout):
    """
    Produce a summary report for the platform set, including
    a breakdown of SLOC per platform subset, code divergence, etc.

    Parameters
    ----------
    setmap: dict[frozenset[str], int]
        The setmap used to compute the summary report.

    stream: TextIO, default: sys.stdout
        The stream to write the report to.
    """
    lines = ["", _heading("Summary", stream)]

    total = sum(setmap.values())
    data = []
    total_count = 0
    for pset in sorted(setmap.keys(), key=len):
        name = "{" + ", ".join(sorted(pset)) + "}"
        count = setmap[pset]
        percent = (float(setmap[pset]) / float(total)) * 100
        data += [[name, str(count), f"{percent:.2f}"]]
        total_count += setmap[pset]

    lines += [
        tabulate(
            data,
            headers=["Platform Set", "LOC", "% LOC"],
            tablefmt="simple_grid",
            floatfmt=".2f",
            stralign="right",
        ),
    ]

    cd = divergence(setmap)
    cc = coverage(setmap)
    ac = average_coverage(setmap)
    lines += [f"Code Divergence: {cd:.2f}"]
    lines += [f"Coverage (%): {cc:.2f}"]
    lines += [f"Avg. Coverage (%): {ac:.2f}"]
    lines += [f"Total SLOC: {total_count}"]

    print("\n".join(lines), file=stream)


def clustering(
    output_name: str,
    setmap: dict[frozenset[str], int],
    stream: TextIO = sys.stdout,
):
    """
    Produce a clustering report for the platform set.

    Parameters
    ----------
    output_name: str
        The filename for the dendrogram.

    setmap: dict[frozenset[str], int]
        The setmap used to compute the clustering statistics.

    stream: TextIO, default: sys.stdout
        The stream to write the report to.
    """
    lines = ["", _heading("Clustering", stream)]

    # Sort the platform list to ensure that the ordering of platforms in the
    # distance matrix and dendrogram do not change from run to run
    platforms = sorted(extract_platforms(setmap))

    if len(platforms) == 1:
        log.error("clustering is not supported for a single platform.")
        return None
    util.ensure_ext(output_name, ".png")

    # Import additional modules required by clustering report
    # Force Agg backend to matplotlib to avoid DISPLAY errors
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    # Remove misleading axes
    for axis in ["left", "right", "top"]:
        matplotlib.rcParams["axes.spines." + axis] = False

    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    # Compute distance matrix between platforms
    matrix = [
        [distance(setmap, p1, p2) for p2 in platforms] for p1 in platforms
    ]

    # Print distance matrix as a table
    lines += ["Distance Matrix:"]
    labelled_matrix = [
        [name] + [f"{column:.2f}" for column in matrix[row]]
        for (row, name) in enumerate(platforms)
    ]
    lines += [
        tabulate(
            labelled_matrix,
            headers=platforms,
            tablefmt="simple_grid",
            floatfmt=".2f",
        ),
    ]

    # Hierarchical clustering using average inter-cluster distance
    clusters = hierarchy.linkage(squareform(matrix), method="average")

    # Plot dendrogram of hierarchical clustering
    # Ignore SciPy warning about axis limits, because we override them
    fig, ax = plt.subplots()
    with warnings.catch_warnings(action="ignore"):
        hierarchy.dendrogram(clusters, labels=platforms, orientation="right")
    ax.set_xlim(xmin=0, xmax=1)
    ax.axvline(x=divergence(setmap), linestyle="--", label="Average")
    ax.text(
        divergence(setmap),
        ax.get_ylim()[1],
        "Average",
        ha="center",
        va="bottom",
    )
    ax.set_xlabel("Code Divergence")
    with util.safe_open_write_binary(output_name) as fp:
        fig.savefig(fp)

    lines += [""]
    lines += [f"Dendrogram written to {output_name}"]

    print("\n".join(lines), file=stream)


def find_duplicates(codebase: CodeBase) -> list[set[Path]]:
    """
    Search for duplicate files in the code base.

    Returns
    -------
    list[set[Path]]
        A list of all sets of Paths with identical contents.
    """
    # Search for possible matches using a hash, ignoring symlinks.
    possible_matches: dict[str, set] = {}
    for path in codebase:
        path = Path(path)
        if path.is_symlink():
            continue
        with open(path, "rb") as f:
            digest = hashlib.file_digest(f, "sha512").hexdigest()
        if digest not in possible_matches:
            possible_matches[digest] = set()
        possible_matches[digest].add(path)

    # Confirm equality for files with the same hash.
    confirmed_matches = []
    for digest, path_set in possible_matches.items():
        # Skip files with no hash conflicts.
        if len(path_set) == 1:
            continue

        # Check for equality amongst all files in the set.
        # Iterate until we have identified all conflicting hashes.
        remaining = path_set.copy()
        while len(remaining) > 1:
            first = remaining.pop()
            matches = {first}
            for path in remaining:
                if filecmp.cmp(first, path, shallow=False):
                    matches.add(path)
            remaining.difference_update(matches)
            if len(matches) > 1:
                confirmed_matches.append(matches)

    return confirmed_matches


def duplicates(codebase: CodeBase, stream: TextIO = sys.stdout):
    """
    Produce a report identifying sets of duplicate files.

    Parameters
    ----------
    codebase: CodeBase
        The code base to search for duplicates.

    stream: TextIO, default: sys.stdout
        The stream to write the report to.
    """
    confirmed_matches = find_duplicates(codebase)

    print("", file=stream)
    print(_heading("Duplicates", stream), file=stream)

    if len(confirmed_matches) == 0:
        print("No duplicates found.", file=stream)
        return

    for i, matches in enumerate(confirmed_matches):
        print(f"Match {i}:", file=stream)
        for path in matches:
            print(f"- {path}")
        if i != len(confirmed_matches) - 1:
            print("")


def _human_readable(x: int) -> str:
    """
    Parameters
    ----------
    x: int
        An integer.

    Returns
    -------
    str
        A human-readable representation of x.

    Raises
    ------
    TypeError
        If `x` is not an integer.
    """
    if not isinstance(x, numbers.Integral):
        raise TypeError("x must be Integral")

    digits = len(str(x))
    if digits <= 3:
        return str(x)
    elif digits <= 6:
        return f"{x/10**3:.1f}k"
    elif digits <= 9:
        return f"{x/10**6:.1f}M"
    elif digits <= 12:
        return f"{x/10**9:.1f}G"
    return "******"


def _strip_colors(s: str) -> str:
    """
    Strip ASCII color codes from a string.

    Parameters
    ----------
    s: str
        The string to strip color codes from.

    Returns
    -------
    str
        A copy of s with all color codes removed.

    Raises
    ------
    TypeError
        If `s` is not a string.
    """
    if not isinstance(s, str):
        raise TypeError("s must be a string")

    return re.sub(r"\033\[[0-9]*m", "", s)


class FileTree:
    """
    A FileTree represents all of the files in a directory using a tree
    structure, with hierarchical tracking of the platform set used by
    each file and directory.
    """

    class Node:
        """
        A FileTree.Node represents a single file or directory in a FileTree.
        """

        def __init__(
            self,
            path,
            setmap=None,
            is_root=False,
        ):
            self.path = Path(path)
            if setmap is None:
                setmap = defaultdict(int)
            self.setmap = setmap
            self.children = {}
            self.is_root = is_root

        @property
        def name(self):
            if self.is_root:
                return str(self.path)
            return str(self.path.name)

        @property
        def platforms(self):
            return extract_platforms(self.setmap)

        @property
        def sloc(self):
            count = 0
            for k, v in self.setmap.items():
                if len(k) == 0:
                    continue
                count += v
            return count

        def is_dir(self):
            return self.path.is_dir()

        def is_symlink(self):
            return self.path.is_symlink()

        def _platforms_str(
            self,
            all_platforms: set[str],
            labels: Sequence[str] = string.ascii_uppercase,
        ) -> str:
            """
            Parameters
            ----------
            all_platforms: set[str]
                The set of all platforms.

            labels: Sequence[str], default: string.ascii_uppercase
                The labels to use in place of real platform names.

            Returns
            -------
            str
                A string representing the platforms used by this Node.
            """
            output = ""
            for i, platform in enumerate(sorted(all_platforms)):
                if platform in self.platforms:
                    if self.is_symlink():
                        color = "\033[96m"
                    else:
                        color = "\033[33m"
                    value = labels[i]
                else:
                    color = "\033[2m"
                    value = "-"
                output += f"{color}{value}\033[0m"
            return output

        def _sloc_str(self, max_sloc: int) -> str:
            """
            Parameters
            ----------
            max_sloc: int
                The maximum SLOC, used to determine formatting width.

            Returns
            -------
            str
                A string representing the SLOC associated with this Node, with
                human-readable numbers.
            """
            color = ""
            if len(self.platforms) == 0:
                color = "\033[2m"
            elif self.is_symlink():
                color = "\033[96m"

            sloc_len = len(_human_readable(max_sloc))
            sloc = _human_readable(sum(self.setmap.values()))

            return f"{color}{sloc:>{sloc_len}}\033[0m"

        def _coverage_str(self, platforms: set[str]) -> str:
            """
            Returns
            -------
            str
                A string representing code coverage of this Node.
            """
            cc = coverage(self.setmap, platforms)
            color = ""
            if len(self.platforms) == 0:
                color = "\033[2m"
            elif self.is_symlink():
                color = "\033[96m"
            elif cc >= 50:
                color = "\033[32m"
            elif cc < 50:
                color = "\033[35m"
            return f"{color}{cc:6.2f}\033[0m"

        def _average_coverage_str(self, platforms: set[str]) -> str:
            """
            Returns
            -------
            str
                A string representing average coverage of this Node.
            """
            cc = average_coverage(self.setmap, platforms)
            color = ""
            if len(self.platforms) == 0:
                color = "\033[2m"
            elif self.is_symlink():
                color = "\033[96m"
            elif cc >= 50:
                color = "\033[32m"
            elif cc < 50:
                color = "\033[35m"
            return f"{color}{cc:6.2f}\033[0m"

        def _meta_str(self, root: Self) -> str:
            """
            Parameters
            ----------
            root: FileTree.Node
                The root of the FileTree containing this FileTree.Node.

            Returns
            -------
            str
                A string representing meta-information for this FileTree.Node.
            """
            max_sloc = sum(root.setmap.values())
            info = [
                self._platforms_str(root.platforms),
                self._sloc_str(max_sloc),
                self._coverage_str(root.platforms),
                self._average_coverage_str(root.platforms),
            ]
            return "[" + " | ".join(info) + "]"

    def __init__(self, rootdir: str | os.PathLike[str]):
        self.root = FileTree.Node(rootdir, is_root=True)

    def insert(
        self,
        filename: str | os.PathLike[str],
        setmap: dict[frozenset[str], int],
    ):
        """
        Insert a new file into the tree, creating as many nodes as necessary.

        Parameters
        ----------
        filename: str | os.PathLike[str]
            The filename to add to the tree.

        setmap: dict | None
            The setmap information associated with this filename.
        """
        rootpath = self.root.path
        filepath = Path(filename)

        parent = self.root
        for path in list(reversed(filepath.parents)) + [filepath]:
            # Do not create nodes above the chosen root directory.
            if path == rootpath or not path.is_relative_to(rootpath):
                continue

            # Do not propagate information from symlinks.
            if not filepath.is_symlink():
                for ps in setmap.keys():
                    parent.setmap[ps] += setmap[ps]

            # If this name exists, find the node.
            if path.name in parent.children:
                node = parent.children[path.name]

            # Otherwise, create the node.
            else:
                if not path.is_dir():
                    node = FileTree.Node(path, setmap)
                else:
                    node = FileTree.Node(path)
                parent.children[path.name] = node

            parent = node

    def _print(
        self,
        node: Node,
        depth: int = 0,
        prefix: str = "",
        connector: str = "",
        fancy: bool = True,
        levels: int | None = None,
    ):
        """
        Recursive helper function to print all nodes in a FileTree.

        Parameters
        ----------
        node: Node
            The current FileTree.Node to print.

        prefix: str, default: ""
            The current prefix, built up from previous recursive calls.

        connector: str, default: ""
            The character used to connect this Node to the FileTree.

        fancy: bool, default: True
            Whether to use fancy formatting (including colors).

        levels: int, optional
            The maximum number of levels to print.
        """
        # Skip this node and its children if we have hit the maximum depth.
        if levels and depth > levels:
            return []

        if fancy:
            dash = "\u2500"
            cont = "\u251C"
            last = "\u2514"
            vert = "\u2502"
        else:
            dash = "-"
            cont = "|"
            last = "\\"
            vert = "|"

        lines = []

        name = ""
        if node.is_dir():
            name += "\033[94m"
        elif node.is_symlink():
            name += "\033[96m"
        elif len(node.platforms) == 0:
            name += "\033[2m"
        name += node.name
        if node.is_dir():
            name += os.sep
        if node.is_symlink():
            name += "\033[0m"
            name += " -> " + str(node.path.resolve())

        name += "\033[0m"

        meta = node._meta_str(self.root)

        # Print this node.
        stub = "" if node == self.root else dash
        stub += "o" if node.is_dir() else dash
        lines += [f"{meta} {prefix}{connector}{stub} {name}"]

        # Prefix children with spaces or vertical line, depending on position.
        if connector == "":
            next_prefix = ""
        elif connector == last:
            next_prefix = prefix + "  "
        else:
            next_prefix = prefix + vert + " "

        for i, name in enumerate(node.children):
            # Use a different connector for the last child in each directory.
            if i == len(node.children) - 1:
                next_connector = last
            else:
                next_connector = cont
            lines += self._print(
                node.children[name],
                depth + 1,
                next_prefix,
                next_connector,
                fancy,
                levels,
            )

        return lines

    def write_to(self, stream: TextIO, levels: int | None = None):
        """
        Write the FileTree to the specified stream.

        Parameters
        ----------
        stream: TextIO
            The text stream to write to.

        levels: int, optional
            The maximum number of levels to print.
            If 0, print only the top-level summary.
        """
        lines = self._print(self.root, fancy=stream.isatty(), levels=levels)
        output = "\n".join(lines)
        if not stream.isatty():
            output = _strip_colors(output)
        print(output, file=stream)


def files(
    codebase: CodeBase,
    state: ParserState | None = None,
    *,
    stream: TextIO = sys.stdout,
    prune: bool = False,
    levels: int | None = None,
):
    """
    Produce a file tree representing the code base.

    Parameters
    ----------
    codebase: CodeBase
        The code base to visualize as a tree.

    state: ParserState, default: None
        An optional ParserState used to annotate the tree with platform
        information.

    stream: TextIO, default: sys.stdout
        The stream to write the report to.

    Raises
    ------
    NotImplementedError
        If the number of directories in the code base is not 1.
    """
    if len(codebase.directories) != 1:
        raise NotImplementedError(
            "'files' report currently requires a code base with 1 directory.",
        )

    # Build up a tree from the list of files.
    tree = FileTree(codebase.directories[0])
    for f in codebase:
        setmap: dict[frozenset[str], int] = defaultdict(int)
        if state:
            association = state.get_map(f)
            for node in filter(
                lambda x: isinstance(x, CodeNode),
                state.get_tree(f).walk(),
            ):
                platform = frozenset(association[node])
                setmap[platform] += node.num_lines
        if prune:
            # Prune unused files from the tree.
            platforms = set().union(*setmap.keys())
            if len(platforms) == 0:
                continue
        tree.insert(f, setmap)

    # Print a legend.
    legend = []
    legend += ["Legend:"]
    for i, platform in enumerate(sorted(tree.root.platforms)):
        label = string.ascii_uppercase[i]
        legend += [f"\033[33m{label}\033[0m: {platform}"]
    legend += [""]
    legend += ["Columns:"]
    header = [
        "Platforms",
        "SLOC",
        "Coverage (%)",
        "Avg. Coverage (%)",
    ]
    legend += ["[" + " | ".join(header) + "]"]
    legend += [""]
    legend_string = "\n".join(legend)
    if not stream.isatty():
        legend_string = _strip_colors(legend_string)
    print(legend_string, file=stream)

    # Print the tree.
    tree.write_to(stream, levels=levels)
