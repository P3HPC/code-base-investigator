# Copyright (C) 2019-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Contains classes and functions for stripping comments and whitespace from
C/C++ files as well as fixed-form Fortran
"""
from __future__ import annotations

import itertools as it
import logging
from collections.abc import Callable, Generator, Iterable, Iterator
from typing import Any, TextIO

from codebasin.language import FileLanguage

log = logging.getLogger(__name__)


class one_space_line:
    """
    A container that represents a single line of code while (generally)
    merging all whitespace into a single space.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.parts: list[str] = []
        self.trailing_space = False

    def append_char(self, c: str) -> None:
        """
        Append a character of no particular class to the line.
        Whitespace will be dropped if the line already ends in space.
        """
        if not c.isspace():
            self.parts.append(c)
            self.trailing_space = False
        else:
            if not self.trailing_space:
                self.parts.append(" ")
                self.trailing_space = True

    def append_space(self) -> None:
        """
        Append whitespace to line, unless line already ends in a space.
        """
        if not self.trailing_space:
            self.parts.append(" ")
            self.trailing_space = True

    def append_nonspace(self, c: str) -> None:
        self.parts.append(c)
        self.trailing_space = False

    def join(self, other: one_space_line) -> None:
        """
        Append another one_space_line to this one, respecting whitespace rules.
        """
        if other.parts:
            if other.parts[0] == " " and self.trailing_space:
                self.parts += other.parts[1:]
            else:
                self.parts += other.parts[:]
            self.trailing_space = other.trailing_space

    def category(self) -> str:
        """
        Report the a category for this line:
        * SRC_NONBLANK if it is non-empty/non-whitespace line of code.
        * BLANK if it is empty or only whitespace.
        * CPP_DIRECTIVE it is is a C preprocessor directive.
        """
        res = "SRC_NONBLANK"
        if not self.parts:
            res = "BLANK"
        elif len(self.parts) == 1:
            if self.parts[0] == " ":
                res = "BLANK"
            elif self.parts[0] == "#":
                res = "CPP_DIRECTIVE"
        elif self.parts[:2] == [" ", "#"] or self.parts[0] == "#":
            res = "CPP_DIRECTIVE"
        return res

    def flush(self) -> str:
        """
        Convert the characters to a string and reset the buffer.
        """
        res = "".join(self.parts)
        self.reset()
        return res


class iter_keep1:
    """
    An iterator wrapper that allows a single item to be 'put back'
    and picked up for the next iteration.
    """

    def __init__(self, iterable: Iterable) -> None:
        self.iterator = iter(iterable)
        self.single = None

    def __iter__(self) -> iter_keep1:
        return self

    def __next__(self) -> Any:
        if self.single is not None:
            res, self.single = self.single, None
            return res
        else:
            return next(self.iterator)

    def putback(self, item: Any) -> None:
        """
        Put item into the iterator such that it will be the next
        yielded item.
        """
        if self.single is not None:
            raise RuntimeError(
                "iter_keep1 can only have one item put back at a time!",
            )
        self.single = item


class c_cleaner:
    """
    Approximation of the early stages of a C preprocessor.
    Joins line continuations, merges whitespace, and replaces comments
    with whitespace. State is kept across physical lines and cleared with
    logical_newline.
    """

    def __init__(
        self,
        outbuf: one_space_line,
        directives_only: bool = False,
    ) -> None:
        """
        directives_only has the cleaner only operate on directive lines.
        """
        self.state = ["TOPLEVEL"]
        self.outbuf = outbuf
        self.directives_only = directives_only
        self.iterkeep = iter_keep1("")

    def logical_newline(self) -> None:
        """
        Reset state when a logical newline is found.
        That is, when a newline without continuation.
        """
        if self.state[-1] == "IN_INLINE_COMMENT":
            self.state = ["TOPLEVEL"]
            self.outbuf.append_space()
        elif self.state[-1] == "FOUND_SLASH":
            self.state = ["TOPLEVEL"]
            self.outbuf.append_nonspace("/")
        elif self.state[-1] == "SINGLE_QUOTATION":
            # This probably should give a warning
            self.state = ["TOPLEVEL"]
        elif self.state[-1] == "DOUBLE_QUOTATION":
            # This probably should give a warning
            self.state = ["TOPLEVEL"]
        elif self.state[-1] == "IN_BLOCK_COMMENT_FOUND_STAR":
            self.state.pop()
            if not self.state[-1] == "IN_BLOCK_COMMENT":
                raise RuntimeError(
                    "Inconsistent parser state. Looking for '/' to "
                    + "terminate non-existent block comment.",
                )
        elif self.state[-1] == "CPP_DIRECTIVE":
            self.state = ["TOPLEVEL"]

    def process(self, lineiter: Iterator) -> None:
        """
        Add contents of lineiter to outbuf, stripping as directed.
        """
        state = self.state
        obuf = self.outbuf
        inbuffer = self.iterkeep
        iter_keep1.__init__(inbuffer, lineiter)
        for char in inbuffer:
            if state[-1] == "TOPLEVEL":
                if self.directives_only:
                    if char == "\\":
                        state.append("ESCAPING")
                        obuf.append_nonspace(char)
                    elif char == "#" and obuf.category() == "BLANK":
                        state.append("CPP_DIRECTIVE")
                        obuf.append_nonspace(char)
                    else:
                        obuf.append_char(char)
                else:
                    if char == "\\":
                        state.append("ESCAPING")
                        obuf.append_nonspace(char)
                    elif char == "/":
                        state.append("FOUND_SLASH")
                    elif char == '"':
                        state.append("DOUBLE_QUOTATION")
                        obuf.append_nonspace(char)
                    elif char == "'":
                        state.append("SINGLE_QUOTATION")
                        obuf.append_nonspace(char)
                    elif char == "#" and obuf.category() == "BLANK":
                        state.append("CPP_DIRECTIVE")
                        obuf.append_nonspace(char)
                    else:
                        obuf.append_char(char)
            elif state[-1] == "CPP_DIRECTIVE":
                if char == "\\":
                    state.append("ESCAPING")
                    obuf.append_nonspace(char)
                elif char == "/":
                    state.append("FOUND_SLASH")
                elif char == '"':
                    state.append("DOUBLE_QUOTATION")
                    obuf.append_nonspace(char)
                elif char == "'":
                    state.append("SINGLE_QUOTATION")
                    obuf.append_nonspace(char)
                else:
                    obuf.append_char(char)
            elif state[-1] == "DOUBLE_QUOTATION":
                if char == "\\":
                    state.append("ESCAPING")
                    obuf.append_nonspace(char)
                elif char == '"':
                    state.pop()
                    obuf.append_nonspace(char)
                else:
                    obuf.append_nonspace(char)
            elif state[-1] == "SINGLE_QUOTATION":
                if char == "\\":
                    state.append("ESCAPING")
                    obuf.append_nonspace(char)
                elif char == "/":
                    state.append("FOUND_SLASH")
                elif char == "'":
                    state.pop()
                    obuf.append_nonspace(char)
                else:
                    obuf.append_nonspace(char)
            elif state[-1] == "FOUND_SLASH":
                if char == "/":
                    state.pop()
                    state.append("IN_INLINE_COMMENT")
                elif char == "*":
                    state.pop()
                    state.append("IN_BLOCK_COMMENT")
                else:
                    state.pop()
                    obuf.append_char("/")
                    inbuffer.putback(char)
            elif state[-1] == "IN_BLOCK_COMMENT":
                if char == "*":
                    state.append("IN_BLOCK_COMMENT_FOUND_STAR")
            elif state[-1] == "IN_BLOCK_COMMENT_FOUND_STAR":
                if char == "/":
                    state.pop()
                    if not state[-1] == "IN_BLOCK_COMMENT":
                        raise RuntimeError(
                            "Inconsistent parser state. Looking for '/' to "
                            + "terminate non-existent block comment.",
                        )
                    state.pop()
                    obuf.append_space()
                elif char != "*":
                    state.pop()
                    if not state[-1] == "IN_BLOCK_COMMENT":
                        raise RuntimeError(
                            "Inconsistent parser state. Looking for '*' to "
                            + "terminate non-existent block comment.",
                        )
            elif state[-1] == "ESCAPING":
                obuf.append_nonspace(char)
                state.pop()
            elif state[-1] == "IN_INLINE_COMMENT":
                return
            else:
                raise RuntimeError("Unknown parser state!")


class fortran_cleaner:
    """
    'Cleans' source to remove comments and blanks while preserving
    directives and handling strings and continuations properly.
    Expects to have c defines already processed.
    """

    def __init__(self, outbuf: one_space_line) -> None:
        self.state = ["TOPLEVEL"]
        self.outbuf = outbuf
        self.verify_continue: list[str] = []

    def dir_check(self, inbuffer: iter_keep1) -> None:
        """
        Inspect comment to see if it is in fact, a valid directive,
        which should be preserved.
        """
        found = ["!"]
        for char in inbuffer:
            if char == "$":
                found.append("$")
                for c in found:
                    self.outbuf.append_nonspace(c)
                for c in inbuffer:
                    self.outbuf.append_nonspace(c)
                break
            elif char.isalpha():
                found.append(char)
            else:
                return

    def process(self, lineiter: Iterator) -> None:
        """
        Add contents of lineiter to current line, removing contents and
        handling continuations.
        """
        inbuffer = iter_keep1(lineiter)
        try:
            while True:
                char = next(inbuffer)
                if self.state[-1] == "TOPLEVEL":
                    if char == "\\":
                        self.state.append("ESCAPING")
                        self.outbuf.append_nonspace(char)
                    elif char == "!":
                        self.dir_check(inbuffer)
                        self.state = ["TOPLEVEL"]
                        break
                    elif char == "&":
                        self.verify_continue.append(char)
                        self.state.append("VERIFY_CONTINUE")
                    elif char == '"':
                        self.state.append("DOUBLE_QUOTATION")
                        self.outbuf.append_nonspace(char)
                    elif char == "'":
                        self.state.append("SINGLE_QUOTATION")
                        self.outbuf.append_nonspace(char)
                    else:
                        self.outbuf.append_char(char)
                elif self.state[-1] == "CONTINUING_FROM_SOL":
                    if char.isspace():
                        self.outbuf.append_space()
                    elif char == "&":
                        self.state.pop()
                    elif char == "!":
                        self.dir_check(inbuffer)
                        break
                    else:
                        self.state.pop()
                        inbuffer.putback(char)
                elif self.state[-1] == "DOUBLE_QUOTATION":
                    if char == "\\":
                        self.state.append("ESCAPING")
                        self.outbuf.append_nonspace(char)
                    elif char == '"':
                        self.state.pop()
                        self.outbuf.append_nonspace(char)
                    elif char == "&":
                        self.verify_continue.append(char)
                        self.state.append("VERIFY_CONTINUE")
                    else:
                        self.outbuf.append_nonspace(char)
                elif self.state[-1] == "SINGLE_QUOTATION":
                    if char == "\\":
                        self.state.append("ESCAPING")
                        self.outbuf.append_nonspace(char)
                    elif char == "'":
                        self.state.pop()
                        self.outbuf.append_nonspace(char)
                    elif char == "&":
                        self.verify_continue.append(char)
                        self.state.append("VERIFY_CONTINUE")
                    else:
                        self.outbuf.append_nonspace(char)
                elif self.state[-1] == "ESCAPING":
                    self.outbuf.append_nonspace(char)
                    self.state.pop()
                elif self.state[-1] == "VERIFY_CONTINUE":
                    if char == "!" and self.state[-2] == "TOPLEVEL":
                        self.dir_check(inbuffer)
                        break
                    elif not char.isspace():
                        for tmp in self.verify_continue:
                            self.outbuf.append_nonspace(tmp)
                        self.verify_continue = []
                        self.state.pop()
                        inbuffer.putback(char)
                    elif char.isspace():
                        self.verify_continue.append(char)
                else:
                    raise RuntimeError("Unknown parser state")
        except StopIteration:
            pass
        if self.state[-1] == "CONTINUING_TO_EOL":
            self.state[-1] = "CONTINUING_FROM_SOL"
        elif self.state[-1] == "VERIFY_CONTINUE":
            self.verify_continue = []
            self.state[-1] = "CONTINUING_FROM_SOL"


class line_info:
    """
    Reprsents a logical line of code.
    """

    def __init__(self) -> None:
        self.current_logical_line = one_space_line()
        self.current_physical_start: int = 1
        self.current_physical_end: int | None = None
        self.lines: list[int] = []
        self.local_sloc: int = 0
        self.category: str | None = None
        self.flushed_line: str | None = None

    def join(self, other_line: one_space_line) -> None:
        """
        Combine this logical line with another one.
        """
        self.current_logical_line.join(other_line)

    def add_physical_lines(self, lines: list[int]) -> None:
        """
        Add the specified physical lines to this logical line.
        """
        self.lines.extend(lines)
        self.local_sloc += len(lines)

    def add_physical_line(self, line: int) -> None:
        """
        Add the specified physical line to this logical line.
        """
        self.add_physical_lines([line])

    def physical_update(self, physical_line_num: int) -> None:
        """
        Mark end of new physical line.
        """
        self.current_physical_end = physical_line_num
        self.category = self.current_logical_line.category()
        self.flushed_line = self.current_logical_line.flush()

    def physical_reset(self) -> int:
        """
        Prepare for next logical block. Return counted sloc.
        """
        if self.current_physical_end is None:
            raise ValueError("Unexpected current_physical_end.")
        self.current_physical_start = self.current_physical_end
        local_sloc_copy = self.local_sloc
        self.lines = []
        self.local_sloc = 0
        self.flushed_line = None
        return local_sloc_copy

    def phys_interval(self) -> tuple[int, int | None]:
        return (self.current_physical_start, self.current_physical_end)


def c_file_source(
    fp: TextIO,
    *,
    directives_only: bool = False,
) -> Generator[line_info, None, tuple[int, int]]:
    """
    Process file fp in terms of logical (sloc) and physical lines of C code.
    Yield blocks of logical lines of code with physical extents.
    Return total lines at exit.
    directives_only sets up parser to only process directive lines such that
    the output can be fed to another file source (i.e. Fortran).
    """
    current_physical_line = one_space_line()
    cleaner = c_cleaner(current_physical_line, directives_only)

    curr_line = line_info()

    total_sloc = 0

    physical_line_num = 0
    continued = False
    for physical_line_num, line in enumerate(fp, start=1):
        current_physical_line.reset()
        end = len(line)
        if line[-1] == "\n":
            end -= 1
        elif end > 0 and line[end - 1] == "\\":
            raise RuntimeError("file seems to end in \\ with no newline!")

        continued = end > 0 and line[end - 1] == "\\"
        if continued:
            end -= 1
        cleaner.process(it.islice(line, 0, end))
        if not continued and cleaner.state[-1] != "IN_BLOCK_COMMENT":
            cleaner.logical_newline()

        if not current_physical_line.category() == "BLANK":
            curr_line.add_physical_line(physical_line_num)

        curr_line.join(current_physical_line)

        if not continued and cleaner.state[-1] != "IN_BLOCK_COMMENT":
            curr_line.physical_update(physical_line_num + 1)
            if curr_line.category != "BLANK":
                yield curr_line

            total_sloc += curr_line.physical_reset()

    total_physical_lines = physical_line_num

    curr_line.physical_update(physical_line_num + 1)
    if curr_line.category != "BLANK":
        yield curr_line

    total_sloc += curr_line.physical_reset()

    # Even if code is technically wrong, we should only fail when necessary.
    parsing_failed = not cleaner.state == ["TOPLEVEL"]
    if continued:
        log.warning("backslash-newline at end of file")
        parsing_failed = False

    if parsing_failed:
        raise RuntimeError(
            "Parsing failed. Please open a bug report at: "
            "https://github.com/P3HPC/code-base-investigator/issues/new?template=bug_report.yml",  # noqa: E501
        )

    return (total_sloc, total_physical_lines)


def fortran_file_source(
    fp: TextIO,
) -> Generator[line_info, None, tuple[int, int]]:
    """
    Process file fp in terms of logical (sloc) and physical lines of
    fixed-form  Fortran code.
    Yield blocks of logical lines of code with physical extents.
    Return total lines at exit.
    """

    current_physical_line = one_space_line()
    cleaner = fortran_cleaner(current_physical_line)

    curr_line = line_info()

    current_physical_start = None
    total_sloc = 0

    c_walker = c_file_source(fp, directives_only=True)
    try:
        while True:
            src_c_line = next(c_walker)
            # If this is a preprocessor directive:
            # - Flush what we have so far
            # - Emit the directive
            # - Start over
            if current_physical_start is None:
                current_physical_start = curr_line.current_physical_start

            if src_c_line.category == "CPP_DIRECTIVE":
                if src_c_line.current_physical_end is None:
                    raise ValueError("Unexpected current_physical_end.")
                curr_line.physical_update(src_c_line.current_physical_end)
                if curr_line.category != "BLANK":
                    yield curr_line

                current_physical_start = None
                total_sloc += curr_line.physical_reset()
                yield src_c_line
                total_sloc += src_c_line.local_sloc
                continue

            current_physical_line.reset()

            if src_c_line.flushed_line is None:
                raise ValueError("Unexpected flushed_line.")
            cleaner.process(
                it.islice(
                    src_c_line.flushed_line,
                    0,
                    len(src_c_line.flushed_line),
                ),
            )

            if not current_physical_line.category() == "BLANK":
                curr_line.add_physical_lines(src_c_line.lines)

            curr_line.join(current_physical_line)

            if cleaner.state[-1] != "CONTINUING_FROM_SOL":
                curr_line.current_physical_start = current_physical_start
                if src_c_line.current_physical_end is None:
                    raise ValueError("Unexpected current_physical_end.")
                curr_line.physical_update(src_c_line.current_physical_end)
                if curr_line.category != "BLANK":
                    yield curr_line

                current_physical_start = None
                total_sloc += curr_line.physical_reset()

    except StopIteration as stopit:
        _, total_physical_lines = stopit.value

    curr_line.physical_update(total_physical_lines)
    if not curr_line.category == "BLANK":
        if current_physical_start is None:
            raise ValueError("Unexpected current_physical_start.")
        curr_line.current_physical_start = current_physical_start
        yield curr_line

    total_sloc += curr_line.physical_reset()

    parsing_failed = not cleaner.state == ["TOPLEVEL"]
    if parsing_failed:
        raise RuntimeError(
            "Parsing failed. Please open a bug report at: "
            "https://github.com/P3HPC/code-base-investigator/issues/new?template=bug_report.yml",  # noqa: E501
        )

    return (total_sloc, total_physical_lines)


class asm_cleaner:
    """
    'Cleans' source to remove comments and blanks while preserving
    directives and handling strings and continuations properly.
    Expects to have c defines already processed.
    """

    def __init__(self, outbuf: one_space_line) -> None:
        self.state = ["TOPLEVEL"]
        self.outbuf = outbuf

    def process(self, lineiter: Iterator) -> None:
        """
        Add contents of lineiter to current line
        """
        inbuffer = iter_keep1(lineiter)
        try:
            while True:
                char = next(inbuffer)

                if self.state[-1] == "TOPLEVEL":
                    if char in ";#":
                        self.outbuf.append_space()
                        return
                    elif char == "/":
                        self.state.append("FOUND_SLASH")
                    else:
                        self.outbuf.append_char(char)
                elif self.state[-1] == "FOUND_SLASH":
                    if char == "/":
                        self.state.pop()
                        self.outbuf.append_space()
                        return
                    else:
                        self.state.pop()
                        self.outbuf.append_char("/")
                        inbuffer.putback(char)
        except StopIteration:
            pass


def asm_file_source(fp: TextIO) -> Generator[line_info, None, tuple[int, int]]:
    """
    Process file fp in terms of logical (sloc) and physical lines of ASM code.
    Yield blocks of logical lines of code with physical extents.
    Return total lines at exit.
    Does not understand NASM-style %if directives
    """
    current_physical_line = one_space_line()
    cleaner = asm_cleaner(current_physical_line)

    curr_line = line_info()

    total_sloc = 0

    physical_line_num = 0
    for physical_line_num, line in enumerate(fp, start=1):
        current_physical_line.reset()
        end = len(line)
        if line[-1] == "\n":
            end -= 1
        cleaner.process(it.islice(line, 0, end))

        if not current_physical_line.category() == "BLANK":
            curr_line.add_physical_line(physical_line_num)

        curr_line.join(current_physical_line)

        curr_line.physical_update(physical_line_num + 1)
        if curr_line.category != "BLANK":
            yield curr_line

        total_sloc += curr_line.physical_reset()

    total_physical_lines = physical_line_num

    curr_line.physical_update(physical_line_num + 1)
    if curr_line.category != "BLANK":
        yield curr_line

    total_sloc += curr_line.physical_reset()

    return (total_sloc, total_physical_lines)


# FIXME: The return type of this function suggests it is too complicated.
def get_file_source(
    path: str,
    assumed_lang: str | None = None,
) -> (
    Callable[[TextIO], Generator[line_info, None, tuple[int, int]]]
    | Callable[[TextIO, bool], Generator[line_info, None, tuple[int, int]]]
):
    """
    Return a C or Fortran line source for path depending on
    the language we can detect, or fail.
    """
    lang = FileLanguage(path).get_language()
    if assumed_lang:
        lang = assumed_lang

    if lang == "fortran-free":
        return fortran_file_source
    elif lang in ["c", "c++"]:
        return c_file_source
    elif lang in ["asm"]:
        return asm_file_source
    else:
        raise RuntimeError(f"Could not determine language of {path}.")
