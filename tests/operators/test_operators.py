# Copyright (C) 2019-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
import unittest
from pathlib import Path

from codebasin import CodeBase, finder, preprocessor
from codebasin.preprocessor import Preprocessor


class TestOperators(unittest.TestCase):
    """
    Simple test of ability to recognize different operators when used
    within directives
    """

    def setUp(self):
        self.rootdir = Path(__file__).parent.resolve()
        logging.disable()

        self.expected_setmap = {frozenset(["CPU", "GPU"]): 32}

    def test_operators(self):
        """operators/operators.yaml"""
        codebase = CodeBase(self.rootdir)
        configuration = {
            "CPU": [
                {
                    "file": str(self.rootdir / "main.cpp"),
                    "defines": ["CPU"],
                    "include_paths": [],
                    "include_files": [],
                },
            ],
            "GPU": [
                {
                    "file": str(self.rootdir / "main.cpp"),
                    "defines": ["GPU"],
                    "include_paths": [],
                    "include_files": [],
                },
            ],
        }
        state = finder.find(self.rootdir, codebase, configuration)
        setmap = state.get_setmap(codebase)
        self.assertDictEqual(
            setmap,
            self.expected_setmap,
            "Mismatch in setmap",
        )

    def test_paths(self):
        input_str = r"FUNCTION(looks/2like/a/path/with_/bad%%identifiers)"
        tokens = preprocessor.Lexer(input_str).tokenize()
        p = Preprocessor(platform_name="Test")
        macro = preprocessor.macro_from_definition_string("FUNCTION(x)=#x")
        p._definitions = {macro.name: macro}
        _ = preprocessor.MacroExpander(p).expand(tokens)


if __name__ == "__main__":
    unittest.main()
