# Copyright (C) 2019-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
import unittest
from pathlib import Path

import codebasin
from codebasin.preprocessor import Identifier, Preprocessor


class TestPreprocessor(unittest.TestCase):
    """
    Test Preprocessor class.
    """

    @classmethod
    def setUpClass(self):
        logging.disable()

    def test_constructor(self):
        """Check arguments are handled correctly"""
        preprocessor = Preprocessor(
            platform_name="name",
            include_paths=["/path/to/include"],
            defines=["MACRO"],
        )
        self.assertEqual(preprocessor.platform_name, "name")
        self.assertCountEqual(
            preprocessor._include_paths,
            [Path("/path/to/include")],
        )

        macro = codebasin.preprocessor.macro_from_definition_string("MACRO")
        self.assertCountEqual(preprocessor._definitions, {"MACRO": macro})

    def test_constructor_validation(self):
        """Check arguments are valid"""

        with self.assertRaises(TypeError):
            Preprocessor(platform_name=1)

        with self.assertRaises(TypeError):
            Preprocessor(include_paths="/not/a/list")

        with self.assertRaises(TypeError):
            Preprocessor(defines="/not/a/list")

    def test_define(self):
        """Check implementation of define"""
        macro = codebasin.preprocessor.macro_from_definition_string("MACRO=x")
        identifier = Identifier("Unknown", -1, False, "MACRO")

        preprocessor = Preprocessor()
        self.assertFalse(preprocessor.has_macro(identifier))
        self.assertIsNone(preprocessor.get_macro(identifier))

        preprocessor.define(macro)
        self.assertTrue(preprocessor.has_macro(identifier))
        self.assertEqual(preprocessor.get_macro(identifier), macro)

    def test_undefine(self):
        """Check implementation of undefine"""
        macro = codebasin.preprocessor.macro_from_definition_string("MACRO=x")
        identifier = Identifier("Unknown", -1, False, "MACRO")

        preprocessor = Preprocessor()
        preprocessor.define(macro)
        preprocessor.undefine(identifier)

        self.assertFalse(preprocessor.has_macro(identifier))
        self.assertIsNone(preprocessor.get_macro(identifier))

    def test_get_file_info(self):
        """Check implementation of get_file_info"""
        preprocessor = Preprocessor()

        info = preprocessor.get_file_info("filename")
        self.assertFalse(info.is_include_once)

        preprocessor.get_file_info("filename").is_include_once = True
        info = preprocessor.get_file_info("filename")
        self.assertTrue(info.is_include_once)


if __name__ == "__main__":
    unittest.main()
