# Copyright (C) 2019-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Contains the Platform class used to specify definitions and include
options for a specific platform.
"""

import os

from codebasin.preprocessor import Macro, MacroFunction


class Platform:
    """
    Represents a platform, and everything associated with a platform.
    Contains a list of definitions, and include paths.
    """

    def __init__(self, name: str, _root_dir: str) -> None:
        self._definitions: dict[str, Macro | MacroFunction] = {}
        self._skip_includes: list[str] = []
        self._include_paths: list[str] = []
        self._root_dir = _root_dir
        self.name = name
        self.found_incl: dict[str, str | None] = {}

    def add_include_path(self, path: str) -> None:
        """
        Insert a new path into the list of include paths for this
        platform.
        """
        self._include_paths.append(path)

    def undefine(self, identifier: str) -> None:
        """
        Undefine a macro for this platform, if it's defined.
        """
        if identifier in self._definitions:
            del self._definitions[identifier]

    def define(self, identifier: str, macro: Macro | MacroFunction) -> None:
        """
        Define a new macro for this platform, only if it's not already
        defined.
        """
        if identifier not in self._definitions:
            self._definitions[identifier] = macro

    def add_include_to_skip(self, fn: str) -> None:
        """
        Define a new macro for this platform, only if it's not already
        defined.
        """
        if fn not in self._skip_includes:
            self._skip_includes.append(fn)

    def process_include(self, fn: str) -> bool:
        """
        Return a boolean stating if this include file should be
        processed or skipped.
        """
        return fn not in self._skip_includes

    # FIXME: This should return a bool, but the usage relies on a str.
    def is_defined(self, identifier: str) -> str:
        """
        Return a string representing whether the macro named by 'identifier' is
        defined.
        """
        if identifier in self._definitions:
            return "1"
        return "0"

    def get_macro(self, identifier: str) -> Macro | MacroFunction | None:
        """
        Return either a macro definition (if it's defined), or None.
        """
        if identifier in self._definitions:
            return self._definitions[identifier]
        return None

    def find_include_file(
        self,
        filename: str,
        this_path: str,
        is_system_include: bool = False,
    ) -> str | None:
        """
        Determine and return the full path to an include file, named
        'filename' using the include paths for this platform.

        System includes do not include the rootdir, while local includes
        do.
        """
        try:
            return self.found_incl[filename]
        except KeyError:
            pass

        include_file = None

        local_paths = []
        if not is_system_include:
            local_paths += [this_path]

        # Determine the path to the include file, if it exists
        for path in local_paths + self._include_paths:
            test_path = os.path.abspath(os.path.join(path, filename))
            if os.path.isfile(test_path):
                include_file = test_path
                self.found_incl[filename] = include_file
                return include_file

        # TODO: Check this optimization is always valid.
        if include_file is not None:
            raise RuntimeError("Expected 'None', got '{filename}'")
        self.found_incl[filename] = None
        return None
