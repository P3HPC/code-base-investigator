# Copyright (C) 2019-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Contains classes and functions related to language detection
and providing information about the language to other parts of
code base investigator
"""

import logging
import os

log = logging.getLogger(__name__)


class FileLanguage:
    """
    Represents the language and modifiers for a given filename
    """

    _supported_languages = ["fortran-free", "fortran-fixed", "c", "c++", "asm"]

    _language_extensions = {}
    _language_extensions["fortran-free"] = [".f90", ".F90"]
    _language_extensions["fortran-fixed"] = [
        ".f",
        ".ftn",
        ".fpp",
        ".F",
        ".FOR",
        ".FTN",
        ".FPP",
    ]
    _language_extensions["c"] = [".c", ".h"]
    _language_extensions["c++"] = [
        ".c++",
        ".cxx",
        ".cpp",
        ".cc",
        ".hpp",
        ".hxx",
        ".h++",
        ".hh",
        ".inc",
        ".inl",
        ".tcc",
        ".icc",
        ".ipp",
        ".cu",
        ".cuh",
        ".cl",
    ]
    _language_extensions["asm"] = [".s", ".S", ".asm"]

    def __init__(self, filename: str):
        self._filename = filename
        self._extension = os.path.splitext(self._filename)[1]
        self._language = None

        for lang in self._supported_languages:
            if self._extension in self._language_extensions[lang]:
                self._language = lang
                break

    def get_language(self) -> str | None:
        return self._language
