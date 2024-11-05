# Copyright (C) 2019-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Contains functions and classes related to finding
and parsing source files as part of a code base.
"""

import collections
import logging
import os
from pathlib import Path

from tqdm import tqdm

from codebasin import CodeBase, file_parser, platform, preprocessor
from codebasin.language import FileLanguage
from codebasin.platform import Platform
from codebasin.preprocessor import CodeNode, Node, Visit

log = logging.getLogger(__name__)


class ParserState:
    """
    Keeps track of the overall state of the parser.
    Contains all of the SourceTree objects created from parsing the
    source files, along with association maps, that associate nodes to
    platforms.
    """

    def __init__(self, summarize_only):
        self.trees = {}
        self.maps = {}
        self.langs = {}
        self.summarize_only = summarize_only
        self._path_cache = {}

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

    def insert_file(self, fn, language=None):
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

    def get_filenames(self):
        """
        Return all of the filenames for files parsed so far.
        """
        return self.trees.keys()

    def get_tree(self, fn):
        """
        Return the SourceTree associated with a filename
        """
        fn = self._get_realpath(fn)
        if fn not in self.trees:
            return None
        return self.trees[fn]

    def get_map(self, fn):
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
        setmap = collections.defaultdict(int)
        for fn in codebase:
            tree = self.get_tree(fn)
            association = self.get_map(fn)
            for node in [n for n in tree.walk() if isinstance(n, CodeNode)]:
                platform = frozenset(association[node])
                setmap[platform] += node.num_lines
        return setmap

    def associate(self, filename: str, platform: Platform):
        """
        Update the association for the provided filename and platform.
        """
        tree = self.get_tree(filename)
        association = self.get_map(filename)
        branch_taken = []

        def associator(node: Node) -> Visit:
            association[node].add(platform.name)
            active = node.evaluate_for_platform(
                platform=platform,
                filename=filename,
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


def find(
    rootdir,
    codebase,
    configuration,
    *,
    summarize_only=True,
    legacy_warnings=True,
    show_progress=False,
):
    """
    Find codepaths in the files provided and return a mapping of source
    lines to platforms.
    """

    # Ensure rootdir is a string for compatibility with legacy code.
    # TODO: Remove this once all other functionality is ported to Path.
    if isinstance(rootdir, Path):
        rootdir = str(rootdir)

    # Build a tree for each unique file for all platforms.
    state = ParserState(summarize_only)
    filenames = set(codebase)
    for p in configuration:
        for e in configuration[p]:
            if e["file"] not in codebase:
                filename = e["file"]
                if legacy_warnings:
                    log.warning(
                        f"{filename} found in definition of platform {p} "
                        + "but missing from codebase",
                    )
            filenames.add(e["file"])
    for f in tqdm(
        filenames,
        desc="Parsing",
        unit=" file",
        leave=False,
        disable=not show_progress,
    ):
        state.insert_file(f)

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
            file_platform = platform.Platform(p, rootdir)

            for path in e["include_paths"]:
                file_platform.add_include_path(path)

            for definition in e["defines"]:
                macro = preprocessor.macro_from_definition_string(definition)
                file_platform.define(macro.name, macro)

            # Process include files.
            # These modify the file_platform instance, but we throw away
            # the active nodes after processing is complete.
            for include in e["include_files"]:
                include_file = file_platform.find_include_file(
                    include,
                    os.path.dirname(e["file"]),
                )
                if include_file:
                    state.insert_file(include_file)
                    state.associate(include_file, file_platform)

            # Process the file, to build a list of associate nodes
            state.associate(e["file"], file_platform)

    return state
