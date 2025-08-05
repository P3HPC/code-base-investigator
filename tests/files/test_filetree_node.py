# Copyright (C) 2019-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from codebasin.report import FileTree, average_coverage, coverage


class TestFileTreeNode(unittest.TestCase):
    """
    Test FileTree.Node functionality.
    """

    @classmethod
    def setUpClass(self):
        logging.disable()

        self.setmap = {
            frozenset(["X"]): 1,
            frozenset(["Y"]): 2,
            frozenset(["X", "Y"]): 3,
            frozenset([]): 6,
        }

        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        open(self.path / "file.cpp", mode="w").close()
        os.symlink(self.path / "file.cpp", self.path / "symlink.cpp")

    @classmethod
    def tearDownClass(self):
        self.tmp.cleanup()

    def test_constructor(self):
        """Check FileTree.Node constructor."""
        node = FileTree.Node(self.path)
        self.assertEqual(node.path, self.path)
        self.assertEqual(node.setmap, defaultdict(int))
        self.assertEqual(node.children, dict())
        self.assertFalse(node.is_root)
        self.assertEqual(node.name, self.path.name)
        self.assertEqual(node.platforms, set())
        self.assertEqual(node.sloc, 0)
        self.assertTrue(node.is_dir())
        self.assertFalse(node.is_symlink())

        node = FileTree.Node(self.path / "file.cpp")
        self.assertFalse(node.is_dir())

        node = FileTree.Node(self.path / "symlink.cpp")
        self.assertFalse(node.is_dir())
        self.assertTrue(node.is_symlink())

        node = FileTree.Node(self.path, setmap=self.setmap, is_root=True)
        self.assertEqual(node.setmap, self.setmap)
        self.assertTrue(node.is_root)
        self.assertEqual(node.name, str(self.path))
        self.assertCountEqual(node.platforms, ["X", "Y"])
        self.assertEqual(node.sloc, 6)

    def test_platforms_str(self):
        """Check platform string format."""
        node = FileTree.Node(self.path / "file.cpp", setmap=self.setmap)
        s = node._platforms_str({"X", "Y"})
        self.assertEqual(s, "\033[33mA\033[0m\033[33mB\033[0m")

        node = FileTree.Node(self.path / "file.cpp")
        s = node._platforms_str({"X", "Y"})
        self.assertEqual(s, "\033[2m-\033[0m\033[2m-\033[0m")

        node = FileTree.Node(self.path / "symlink.cpp", setmap=self.setmap)
        s = node._platforms_str({"X", "Y"})
        self.assertEqual(s, "\033[96mA\033[0m\033[96mB\033[0m")

        node = FileTree.Node(
            self.path / "symlink.cpp",
            setmap=defaultdict(int),
        )
        s = node._platforms_str({"X", "Y"})
        self.assertEqual(s, "\033[2m-\033[0m\033[2m-\033[0m")

        node = FileTree.Node(self.path / "file.cpp", setmap=self.setmap)
        s = node._platforms_str({"X", "Y"}, labels=["Q", "R"])
        self.assertEqual(s, "\033[33mQ\033[0m\033[33mR\033[0m")

    def test_sloc_str(self):
        """Check SLOC string format."""
        node = FileTree.Node(self.path / "file.cpp", setmap=self.setmap)
        s = node._sloc_str(12)
        self.assertEqual(s, "12\033[0m")

        s = node._sloc_str(1000)
        self.assertEqual(s, "  12\033[0m")

        node = FileTree.Node(self.path / "file.cpp")
        s = node._sloc_str(12)
        self.assertEqual(s, "\033[2m 0\033[0m")

        node = FileTree.Node(self.path / "symlink.cpp", setmap=self.setmap)
        s = node._sloc_str(12)
        self.assertEqual(s, "\033[96m12\033[0m")

        node = FileTree.Node(
            self.path / "symlink.cpp",
            setmap=defaultdict(int),
        )
        s = node._sloc_str(12)
        self.assertEqual(s, "\033[2m 0\033[0m")

    def test_coverage_str(self):
        """Check coverage string format."""
        node = FileTree.Node(self.path / "file.cpp", setmap=self.setmap)
        cc = coverage(self.setmap, ["X", "Y"])
        s = node._coverage_str(["X", "Y"])
        self.assertEqual(s, f"\033[32m{cc:6.2f}\033[0m")

        bad_setmap = {
            frozenset(["X"]): 1,
            frozenset([]): 6,
        }
        node = FileTree.Node(self.path / "file.cpp", setmap=bad_setmap)
        cc = coverage(bad_setmap, ["X", "Y"])
        s = node._coverage_str(["X", "Y"])
        self.assertEqual(s, f"\033[35m{cc:6.2f}\033[0m")

        node = FileTree.Node(self.path / "file.cpp")
        cc = float("nan")
        s = node._coverage_str(["X", "Y"])
        self.assertEqual(s, f"\033[2m{cc:6.2f}\033[0m")

        node = FileTree.Node(self.path / "symlink.cpp", setmap=self.setmap)
        cc = coverage(self.setmap, ["X", "Y"])
        s = node._coverage_str(["X", "Y"])
        self.assertEqual(s, f"\033[96m{cc:6.2f}\033[0m")

        node = FileTree.Node(self.path / "symlink.cpp")
        cc = float("nan")
        s = node._coverage_str(["X", "Y"])
        self.assertEqual(s, f"\033[2m{cc:6.2f}\033[0m")

    def test_average_coverage_str(self):
        """Check average_coverage string format."""
        node = FileTree.Node(self.path / "file.cpp", setmap=self.setmap)
        cc = average_coverage(self.setmap, ["X", "Y"])
        s = node._average_coverage_str(["X", "Y"])
        self.assertEqual(s, f"\033[35m{cc:6.2f}\033[0m")

        bad_setmap = {
            frozenset(["X"]): 1,
            frozenset([]): 6,
        }
        node = FileTree.Node(self.path / "file.cpp", setmap=bad_setmap)
        cc = average_coverage(bad_setmap, ["X", "Y"])
        s = node._average_coverage_str(["X", "Y"])
        self.assertEqual(s, f"\033[35m{cc:6.2f}\033[0m")

        node = FileTree.Node(self.path / "file.cpp")
        cc = float("nan")
        s = node._average_coverage_str(["X", "Y"])
        self.assertEqual(s, f"\033[2m{cc:6.2f}\033[0m")

        node = FileTree.Node(self.path / "symlink.cpp", setmap=self.setmap)
        cc = average_coverage(self.setmap, ["X", "Y"])
        s = node._average_coverage_str(["X", "Y"])
        self.assertEqual(s, f"\033[96m{cc:6.2f}\033[0m")

        node = FileTree.Node(self.path / "symlink.cpp")
        cc = float("nan")
        s = node._average_coverage_str(["X", "Y"])
        self.assertEqual(s, f"\033[2m{cc:6.2f}\033[0m")


if __name__ == "__main__":
    unittest.main()
