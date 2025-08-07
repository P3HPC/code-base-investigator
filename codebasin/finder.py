# Copyright (C) 2019-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Contains functions and classes related to finding
and parsing source files as part of a code base.
"""

import collections
import logging
import os
from collections.abc import Generator, KeysView
from pathlib import Path
from typing import Any

from tqdm import tqdm

from codebasin import CodeBase, file_parser
from codebasin.language import FileLanguage
from codebasin.preprocessor import (
    CodeNode,
    Node,
    Preprocessor,
    SourceTree,
    Visit,
)

log = logging.getLogger(__name__)


class ParserState:
    """
    Keeps track of the overall state of the parser.
    Contains all of the SourceTree objects created from parsing the
    source files, along with association maps, that associate nodes to
    platforms.
    """

    def __init__(self, summarize_only: bool) -> None:
        self.trees: dict[str, SourceTree] = {}
        self.maps: dict[str, dict[Node, set]] = {}
        self.langs: dict[str, str | None] = {}
        self.summarize_only = summarize_only
        self._path_cache: dict[str, str] = {}

    def _get_realpath(self, path: str) -> str:
        """
        Returns
        -------
        str
            Equivalent to os.path.realpath(path).
        """
        if path not in self._path_cache:
            real = os.path.realpath(path)
            self._path_cache[path] = real
        return self._path_cache[path]

    def insert_file(self, fn: str, language: str | None = None) -> None:
        """
        Build a new tree for a source file, and create an association
        map for it.
        """
        fn = self._get_realpath(fn)
        if fn not in self.trees:
            parser = file_parser.FileParser(fn)
            self.trees[fn] = parser.parse_file(
                summarize_only=self.summarize_only,
                language=language,
            )
            self.maps[fn] = collections.defaultdict(set)
            if language:
                self.langs[fn] = language
            else:
                self.langs[fn] = FileLanguage(fn).get_language()

    def get_filenames(self) -> KeysView[str]:
        """
        Return all of the filenames for files parsed so far.
        """
        return self.trees.keys()

    def get_tree(self, fn: str) -> SourceTree | None:
        """
        Return the SourceTree associated with a filename
        """
        fn = self._get_realpath(fn)
        if fn not in self.trees:
            return None
        return self.trees[fn]

    def get_map(self, fn: str) -> dict[Node, set] | None:
        """
        Return the NodeAssociationMap associated with a filename
        """
        fn = self._get_realpath(fn)
        if fn not in self.maps:
            return None
        return self.maps[fn]

    def get_setmap(self, codebase: CodeBase) -> dict[frozenset, int]:
        """
        Returns
        -------
        dict[frozenset, int]
            The number of lines associated with each platform set.
        """
        setmap: dict[frozenset, int] = collections.defaultdict(int)
        for fn in codebase:
            # Don't count symlinks if their target is in the code base.
            # The target will be counted separately.
            path = Path(fn)
            if path.is_symlink() and path.resolve() in codebase:
                continue

            tree = self.get_tree(fn)
            association = self.get_map(fn)
            if tree is None or association is None:
                raise RuntimeError(f"Missing tree or association for '{fn}'")
            for node in [n for n in tree.walk() if isinstance(n, CodeNode)]:
                platform = frozenset(association[node])
                setmap[platform] += node.num_lines
        return setmap

    def associate(self, filename: str, preprocessor: Preprocessor) -> None:
        """
        Update the association for `filename` using `preprocessor`.
        """
        tree = self.get_tree(filename)
        association = self.get_map(filename)
        if tree is None or association is None:
            raise RuntimeError(f"Missing tree or association for '{filename}'")

        if preprocessor.platform_name is None:
            raise RuntimeError(f"Cannot associate '{filename}' with 'None'")

        branch_taken = []

        def associator(node: Node) -> Visit:
            association[node].add(preprocessor.platform_name)

            # TODO: Consider inverting, so preprocessor calls the function.
            active = node.evaluate(
                preprocessor=preprocessor,
                filename=self._get_realpath(filename),
                state=self,
            )

            # Ensure we only descend into one branch of an if/else/endif.
            if node.is_start_node():
                branch_taken.append(active)
            elif node.is_cont_node():
                if branch_taken[-1]:
                    return Visit.NEXT_SIBLING
                branch_taken[-1] = active
            elif node.is_end_node():
                branch_taken.pop()

            if active:
                return Visit.NEXT
            else:
                return Visit.NEXT_SIBLING

        tree.visit(associator)


# FIXME: configuration should be refactored to avoid such a complex type.
def find(
    rootdir: str,
    codebase: CodeBase,
    configuration: dict[Any, list[dict[str, Any]]],
    *,
    summarize_only: bool = True,
    show_progress: bool = False,
) -> ParserState:
    """
    Find codepaths in the files provided and return a mapping of source
    lines to platforms.
    """

    # Ensure rootdir is a string for compatibility with legacy code.
    # TODO: Remove this once all other functionality is ported to Path.
    if isinstance(rootdir, Path):
        rootdir = str(rootdir)

    # Build up a list of potential source files.
    def _potential_file_generator(
        codebase: CodeBase,
    ) -> Generator[Path, None, None]:
        for directory in codebase._directories:
            yield from Path(directory).rglob("*")

    potential_files = []
    for f in tqdm(
        _potential_file_generator(codebase),
        desc="Scanning current directory",
        unit=" files",
        leave=False,
        disable=not show_progress,
    ):
        potential_files.append(f)

    # Identify which files are in the code base.
    filenames = set()
    for f in tqdm(
        potential_files,
        desc="Identifying source files",
        unit=" files",
        leave=False,
        disable=not show_progress,
    ):
        if f in codebase:
            filenames.add(f)
    for p in configuration:
        for e in configuration[p]:
            filenames.add(e["file"])

    # Build a tree for each unique file for all platforms.
    state = ParserState(summarize_only)
    for f in tqdm(
        filenames,
        desc="Parsing",
        unit=" file",
        leave=False,
        disable=not show_progress,
    ):
        log.debug(f"Parsing {f}")
        state.insert_file(str(f))

    # Process each tree, by associating nodes with platforms
    for p in tqdm(
        configuration,
        desc="Preprocessing",
        unit=" platform",
        leave=False,
        disable=not show_progress,
    ):
        for e in tqdm(
            configuration[p],
            desc=p,
            unit=" file",
            leave=False,
            disable=not show_progress,
        ):
            preprocessor = Preprocessor(
                platform_name=p,
                include_paths=e["include_paths"],
                defines=e["defines"],
            )

            # Process include files.
            # These modify the preprocessor instance, but we throw away
            # the active nodes after processing is complete.
            for include in e["include_files"]:
                include_file = preprocessor.find_include_file(
                    include,
                    os.path.dirname(e["file"]),
                )
                if include_file:
                    state.insert_file(include_file)
                    state.associate(include_file, preprocessor)

            # Process the file, to build a list of associate nodes
            # TODO: Consider inverting, so preprocessor calls the function.
            state.associate(e["file"], preprocessor)

    return state
