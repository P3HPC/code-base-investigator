# Copyright (C) 2019-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import importlib.metadata
import os
import shlex
import typing
import warnings
from collections.abc import Generator, Iterable
from pathlib import Path

import pathspec

import codebasin.source
import codebasin.util

warnings.warn(
    "Calling codebasin package internals is deprecated. "
    + "Please call the codebasin script directly instead. "
    + "A new, stable, package interface will be introduced in "
    + "a future release of Code Base Investigator.",
    DeprecationWarning,
)

__version__ = importlib.metadata.version("codebasin")


class CompileCommand:
    """
    A single compile command from a compilation database.

    Attributes
    ----------
    filename: string
        The name of the source file compiled by this command.

    directory: string, optional
        The working directory for this command.

    arguments: list[string], optional
        The `argv` for this command, including the executable as `argv[0]`.

    command: string, optional
        The command as a string.

    output: string, optional
        The name of the file produced by this command, or None if not
        specified.
    """

    def __init__(
        self,
        filename: str,
        directory: str | None = None,
        arguments: list[str] | None = None,
        command: str | None = None,
        output: str | None = None,
    ) -> None:
        """
        Raises
        ------
        ValueError
            If both arguments and command are None.
        """
        self._filename = filename
        self._directory = directory
        if arguments is None and command is None:
            raise ValueError("CompileCommand requires arguments or command.")
        self._arguments = arguments
        self._command = command
        self._output = output

    @property
    def directory(self) -> str | None:
        return self._directory

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def arguments(self) -> list[str]:
        if self._arguments is None:
            if self._command is not None:
                return shlex.split(self._command)
            else:
                raise ValueError(
                    "CompileCommand requires arguments or command.",
                )
        else:
            return self._arguments

    @property
    def output(self) -> str | None:
        return self._output

    def __str__(self) -> str:
        return " ".join(self.arguments)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompileCommand):
            return NotImplemented
        props = ["directory", "filename", "arguments", "output"]
        return all([getattr(self, p) == getattr(other, p) for p in props])

    def is_supported(self) -> bool:
        """
        Returns
        -------
        bool
            True if the command can be emulated and False otherwise.
            Commands that are not supported will not impact analysis.
        """
        # Commands must be non-empty in order to do something.
        # Commands must operate on source files.
        if len(self.arguments) > 0 and codebasin.source.is_source_file(
            self.filename,
        ):
            return True

        return False

    @classmethod
    def from_json(cls, instance: dict) -> CompileCommand:
        """
        Parameters
        ----------
        instance: dict
            A JSON object representing a single compile command.

        Returns
        -------
        CompileCommand
            A CompileCommand corresponding to the JSON object.
        """
        directory = instance.get("directory", None)
        arguments = instance.get("arguments", None)
        command = instance.get("command", None)
        output = instance.get("output", None)
        return cls(
            instance["file"],
            directory=directory,
            arguments=arguments,
            command=command,
            output=output,
        )


class CompilationDatabase:
    """
    A compilation database containing multiple CompileCommands.
    """

    def __init__(self, commands: list[CompileCommand]):
        self.commands = commands

    def __iter__(self) -> Generator[CompileCommand, None, None]:
        """
        Iterate over all commands in the compilation database.
        """
        yield from self.commands

    @classmethod
    def from_json(cls, instance: list) -> CompilationDatabase:
        """
        Parameters
        ----------
        instance: list
            A JSON representation of a list of compile commands.

        Raises
        ------
        ValueError
            If the JSON fails validation.

        Returns
        -------
        CompilationDatabase
            A CompilationDatabase corresponding to the provided JSON.
        """
        codebasin.util._validate_json(instance, "compiledb")
        commands = [CompileCommand.from_json(c) for c in instance]
        return cls(commands)

    @classmethod
    def from_file(
        cls,
        filename: str | os.PathLike[str],
    ) -> CompilationDatabase:
        """
        Parameters
        ---------
        filename: str | os.PathLike[str]
            A JSON file containing a compilation database.

        Raises
        ------
        ValueError
            If the JSON fails validation.

        FileNotFoundError
            If the file with the specified name does not exist.

        Returns
        -------
            A CompilationDatbase corresponding to the provided JSON file.
        """
        with open(filename) as f:
            db: object = codebasin.util._load_json(f, schema_name="compiledb")
        return CompilationDatabase.from_json(typing.cast(list, db))


class CodeBase:
    """
    A representation of all source files in the code base.

    Attributes
    ----------
    directories: list[str | os.PathLike[str]]
        The set of source directories that make up the code base.

    exclude_patterns: list[str]
        A set of patterns describing source files excluded from the code base.
    """

    def __init__(
        self,
        *directories: str | os.PathLike[str],
        exclude_patterns: list[str] = [],
    ):
        """
        Raises
        ------
        TypeError
            If any directory in `directories` is not a path.
            If `exclude_patterns` is not a list of strings.
        """
        if not isinstance(exclude_patterns, list):
            raise TypeError("'exclude_patterns' must be a list.")
        if not all([isinstance(d, (str, os.PathLike)) for d in directories]):
            raise TypeError(
                "Each directory in 'directories' must be PathLike.",
            )
        if not all([isinstance(p, str) for p in exclude_patterns]):
            raise TypeError(
                "Each pattern in 'exclude_patterns' must be a string.",
            )
        self._directories = [Path(d).resolve() for d in directories]
        self._excludes = exclude_patterns

    def __repr__(self) -> str:
        return (
            f"CodeBase(directories={self.directories}, "
            + f"exclude_patterns={self.exclude_patterns})"
        )

    @property
    def directories(self) -> list[str]:
        return [str(d) for d in self._directories]

    @property
    def exclude_patterns(self) -> list[str]:
        return self._excludes

    def __contains__(self, path: os.PathLike) -> bool:
        """
        Returns
        -------
        bool
            True if `path` is a recognized source file in one of the code
            base's listed directories and does not match any exclude
            pattern(s).
        """
        path = Path(path).resolve()

        # Files that don't exist aren't part of the code base.
        if not path.exists():
            return False

        # Directories cannot be source files.
        if path.is_dir():
            return False

        # Files with unrecognized extensions are not source files.
        if not codebasin.source.is_source_file(path):
            return False

        # Files outside of any directory are not in the code base.
        # Store the root for evaluation of relative exclude paths later.
        root = None
        for directory in self.directories:
            if path.is_relative_to(directory):
                root = directory
                break
        if root is None:
            return False

        # Files matching an exclude pattern are not in the code base.
        #
        # Use GitIgnoreSpec to match git behavior in weird corner cases.
        # Convert relative paths to match .gitignore subdirectory behavior.
        spec = pathspec.GitIgnoreSpec.from_lines(self.exclude_patterns)
        try:
            relative_path = path.relative_to(root)
            if spec.match_file(relative_path):
                return False
        except ValueError:
            pass

        return True

    def __iter__(self) -> Generator[str, None, None]:
        """
        Iterate over all files in the code base by walking each directory.
        """
        for directory in self.directories:
            for path in Path(directory).rglob("*"):
                if self.__contains__(path):
                    yield str(path)
