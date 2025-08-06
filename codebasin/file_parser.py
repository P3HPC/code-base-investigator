# Copyright (C) 2019-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Contains classes and functions related to parsing a file,
and building a tree of nodes from it.
"""
from __future__ import annotations

import logging
import os

from codebasin import preprocessor
from codebasin.file_source import get_file_source
from codebasin.preprocessor import SourceTree

log = logging.getLogger(__name__)


class LineGroup:
    """
    Represents a grouping of lines. It contains the extent, and the
    number of countable lines for the group.
    """

    def __init__(self) -> None:
        self.line_count = 0
        self.start_line = -1
        self.end_line = -1
        self.lines: list[int] = []
        self.body: list[str] = []

    def empty(self) -> bool:
        """
        Return a boolean that tells if this group contains any line
        information or not.
        """
        if (
            (self.line_count != 0)
            or (self.start_line != -1)
            or (self.end_line != -1)
        ):
            return False
        return True

    def add_line(
        self,
        phys_int: tuple[int, int],
        sloc_count: int,
        source: str | None = None,
        lines: list[int] | None = None,
    ) -> None:
        """
        Add a line to this line group. Update the extent appropriately,
        and if it's a countable line, add it to the line count.
        """

        if self.start_line == -1 or phys_int[0] < self.start_line:
            self.start_line = phys_int[0]

        if phys_int[1] - 1 > self.end_line:
            self.end_line = phys_int[1] - 1

        self.line_count += sloc_count
        if source is not None:
            self.body.append(source)
        if lines is not None:
            self.lines.extend(lines)

    def reset(self) -> None:
        """
        Reset the countable group
        """
        self.line_count = 0
        self.start_line = -1
        self.end_line = -1
        self.lines = []
        self.body = []

    def merge(self, line_group: LineGroup) -> None:
        """
        Merge another line group into this line group, and reset the
        other group.
        """
        self.line_count += line_group.line_count

        if self.start_line == -1:
            self.start_line = line_group.start_line
        elif line_group.start_line == -1:
            line_group.start_line = self.start_line
        self.start_line = min(self.start_line, line_group.start_line)

        self.lines.extend(line_group.lines)
        self.body.extend(line_group.body)

        self.end_line = max(self.end_line, line_group.end_line)
        line_group.reset()


class FileParser:
    """
    Contains methods for parsing an entire source file and returning a
    source tree object, along with utility methods for determining
    information about source lines and helping to build the source tree.
    """

    def __init__(self, _filename: str):
        self._filename = os.path.abspath(_filename)

    @staticmethod
    def handle_directive(
        out_tree: SourceTree,
        groups: dict[str, LineGroup],
        logical_line: str,
    ) -> None:
        """
        Handle inserting code and directive nodes, where appropriate.
        Update the file group, and reset the code and directive groups.
        """
        # We will actually use this directive, if it is not empty
        # We need to finalize the previously started
        # CodeNode (if there was one) before processing
        # this DirectiveNode
        if not groups["code"].empty():
            FileParser.insert_code_node(out_tree, groups["code"])
            groups["file"].merge(groups["code"])

        FileParser.insert_directive_node(
            out_tree,
            groups["directive"],
            logical_line,
        )

        groups["file"].merge(groups["directive"])

    @staticmethod
    def insert_code_node(tree: SourceTree, line_group: LineGroup) -> None:
        """
        Build a code node, and insert it into the source tree
        """
        new_node = preprocessor.CodeNode(
            line_group.start_line,
            line_group.end_line,
            line_group.line_count,
            line_group.body,
            lines=line_group.lines,
        )
        tree.insert(new_node)

    @staticmethod
    def insert_directive_node(
        tree: SourceTree,
        line_group: LineGroup,
        logical_line: str,
    ) -> None:
        """
        Build a directive node by parsing a directive line, and insert a
        new directive node into the tree.
        """
        new_node = preprocessor.DirectiveParser(
            preprocessor.Lexer(logical_line, line_group.start_line).tokenize(),
        ).parse()
        new_node.start_line = line_group.start_line
        new_node.end_line = line_group.end_line
        new_node.num_lines = line_group.line_count
        new_node.lines = line_group.lines

        # Issue a warning for unrecognized directives, but suppress warnings
        # for common directives that shouldn't impact correctness.
        if isinstance(new_node, preprocessor.UnrecognizedDirectiveNode):
            tokens = new_node.tokens
            unhandled = ["line", "warning", "error"]
            if len(tokens) >= 2 and str(tokens[1]) not in unhandled:
                filename = tree.root.filename
                line = tokens[0].line
                column = tokens[0].col
                spelling = new_node.spelling()
                message = f"unrecognized directive '{spelling}'"
                log.warning(f"{filename}:{line}:{column}: {message}")

        tree.insert(new_node)

    def parse_file(
        self,
        *,
        summarize_only: bool = False,
        language: str | None = None,
    ) -> SourceTree:
        """
        Parse the file that this parser points at, build a SourceTree
        representing this file, and return it.
        """

        filename = self._filename
        out_tree = preprocessor.SourceTree(filename)
        file_source = get_file_source(filename, language)
        if not callable(file_source):
            raise RuntimeError(
                f"{filename} doesn't appear "
                + "to be a language this tool can process",
            )
        with open(filename, errors="replace") as source_file:
            groups = {
                "code": LineGroup(),
                "directive": LineGroup(),
                "file": LineGroup(),
            }

            groups["file"].start_line = 1

            source = file_source(source_file)
            try:
                while True:
                    logical_line = next(source)
                    phys_int = logical_line.phys_interval()
                    # Only follow continuation for directives
                    if logical_line.category == "CPP_DIRECTIVE":
                        # Add this into the directive lines, even if it
                        # might not be a directive we count

                        groups["directive"].add_line(
                            phys_int,
                            logical_line.local_sloc,
                            logical_line.flushed_line,
                            lines=logical_line.lines,
                        )

                        FileParser.handle_directive(
                            out_tree,
                            groups,
                            logical_line.flushed_line,
                        )

                        # FallBack is that this line is a simple code line.
                    else:
                        if summarize_only:
                            groups["code"].add_line(
                                phys_int,
                                logical_line.local_sloc,
                                lines=logical_line.lines,
                            )
                        else:
                            groups["code"].add_line(
                                phys_int,
                                logical_line.local_sloc,
                                logical_line.flushed_line,
                                lines=logical_line.lines,
                            )
            except StopIteration as it:
                _, physical_loc = it.value

            if not groups["code"].empty():
                groups["code"].add_line(
                    (groups["code"].start_line, physical_loc - 1),
                    0,
                )
                self.insert_code_node(out_tree, groups["code"])
                groups["file"].merge(groups["code"])

            out_tree.root.num_lines = groups["file"].end_line
            out_tree.root.total_sloc = groups["file"].line_count
            return out_tree
