# Copyright (C) 2019-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Contains functions to build up a configuration dictionary,
defining a specific code base configuration.
"""
from __future__ import annotations

import argparse
import logging
import os
import pkgutil
import re
import string
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any

from codebasin import CompilationDatabase, util

log = logging.getLogger(__name__)

_compilers = {}


class _StoreSplitAction(argparse.Action):
    """
    A custom argparse.Action that splits the value based on a user-provided
    separator, then stores the resulting list.
    """

    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        nargs=None,
        **kwargs,
    ):
        self.sep = kwargs.pop("sep", None)
        self.format = kwargs.pop("format", None)
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        if not isinstance(values, str):
            raise TypeError("store_split expects string values")
        split_values = values.split(self.sep)
        if self.format:
            template = string.Template(self.format)
            split_values = [template.substitute(value=v) for v in split_values]
        if self.dest == "passes":
            passes = getattr(namespace, "_passes")
            passes[option_string] = split_values
        else:
            setattr(namespace, self.dest, split_values)


class _ExtendMatchAction(argparse.Action):
    """
    A custom argparse.Action that matches the value against a user-provided
    pattern, then extends the destination list using the result(s).
    """

    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        nargs=None,
        **kwargs,
    ):
        self.pattern = kwargs.pop("pattern", None)
        self.format = kwargs.pop("format", None)
        self.override = kwargs.pop("override", False)
        self.flag_name = option_strings[0]
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        value: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        if not isinstance(value, str):
            raise TypeError("extend_match expects string value")
        matches = re.findall(self.pattern, value)
        if self.format:
            template = string.Template(self.format)
            matches = [template.substitute(value=v) for v in matches]
        if self.dest == "passes":
            passes = getattr(namespace, "_passes")
            if self.flag_name not in passes:
                passes[self.flag_name] = []
            if self.override:
                passes[self.flag_name] = matches
                self.override = False
            else:
                passes[self.flag_name].extend(matches)
        else:
            if self.override:
                setattr(namespace, self.dest, matches)
            else:
                dest = getattr(namespace, self.dest)
                dest.extend(matches)


@dataclass
class _CompilerMode:
    name: str
    defines: list[str] = field(default_factory=list)
    include_paths: list[str] = field(default_factory=list)
    include_files: list[str] = field(default_factory=list)

    @classmethod
    def from_toml(cls, toml: dict[str, Any]) -> _CompilerMode:
        return _CompilerMode(**toml)


@dataclass
class _CompilerPass:
    name: str
    defines: list[str] = field(default_factory=list)
    include_paths: list[str] = field(default_factory=list)
    include_files: list[str] = field(default_factory=list)
    modes: list[str] = field(default_factory=list)

    @classmethod
    def from_toml(cls, toml: dict[str, Any]) -> _CompilerPass:
        return _CompilerPass(**toml)


@dataclass
class _Compiler:
    alias_of: str | None = None
    options: list[str] = field(default_factory=list)
    parser: list[dict] = field(default_factory=list)
    modes: dict[str, _CompilerMode] = field(default_factory=dict)
    passes: dict[str, _CompilerPass] = field(default_factory=dict)

    @classmethod
    def from_toml(cls, toml: dict[str, Any]) -> _Compiler:
        kwargs = toml.copy()
        if "parser" in kwargs:
            for option in kwargs["parser"]:
                if option["action"] == "store_split":
                    option["action"] = _StoreSplitAction
                if option["action"] == "extend_match":
                    option["action"] = _ExtendMatchAction
        if "modes" in toml:
            kwargs["modes"] = {
                m["name"]: _CompilerMode.from_toml(m) for m in kwargs["modes"]
            }
        if "passes" in toml:
            kwargs["passes"] = {
                p["name"]: _CompilerPass.from_toml(p) for p in kwargs["passes"]
            }
        return _Compiler(**kwargs)


def _load_compilers():
    """
    Load the configuration from the following files:
    - ${PACKAGE}/compilers/*.toml
    - .cbi/config
    """
    global _compilers
    _compilers = {}

    # Load the package-provided configuration files.
    for compiler in ["clang", "gnu", "intel", "nvidia"]:
        filename = str((Path("compilers") / compiler).with_suffix(".toml"))
        toml = tomllib.loads(
            pkgutil.get_data("codebasin", filename).decode(),
        )
        try:
            util._validate_toml(toml, "cbiconfig")
        except ValueError as e:
            log.error(str(e))
            return

        for name, definition in toml["compiler"].items():
            _compilers[name] = _Compiler.from_toml(definition)

    # Check for any user-defined compiler behavior.
    path = ".cbi/config"
    if os.path.exists(path):
        log.info(f"Found configuration file at {path}.")
        with open(path, "rb") as f:
            try:
                toml = util._load_toml(f, "cbiconfig")
            except ValueError as e:
                log.error(str(e))
                return

        # Ignore empty configuration files.
        if "compiler" not in toml:
            return

        for name, definition in toml["compiler"].items():
            # Check if the user is defining a new compiler.
            if name not in _compilers:
                _compilers[name] = _Compiler.from_toml(definition)
                continue

            compiler = _compilers[name]

            # Check if the user is redefining a compiler as an alias.
            # Warn because options may be lost.
            if "alias_of" in definition:
                log.warning(
                    f'{name} redefined as alias of {definition["alias_of"]}.',
                )
                _compilers[name] = _Compiler.from_toml(definition)
                continue

            # Check if the user is redefining an alias as a compiler.
            # Warn because options may be lost.
            if compiler.alias_of:
                log.warning(
                    f"definition of {name} in .cbi/config overrides alias.",
                )
                compiler.alias_of = None

            # Append options, parser options, modes and passes.
            # Warn if modes/passes are redefined because options may be lost.
            if "options" in definition:
                compiler.options.extend(definition["options"])
            if "parser" in definition:
                for option in definition["parser"]:
                    option = option.copy()
                    if option["action"] == "store_split":
                        option["action"] = _StoreSplitAction
                    if option["action"] == "extend_match":
                        option["action"] = _ExtendMatchAction
                    compiler.parser.append(option)
            if "modes" in definition:
                for m in definition["modes"]:
                    name = m["name"]
                    if name in compiler.modes:
                        log.warning(f"compiler mode '{name}' redefined")
                    compiler.modes[name] = _CompilerMode.from_toml(m)
            if "passes" in definition:
                for p in definition["passes"]:
                    name = p["name"]
                    if name in compiler.passes:
                        log.warning(f"compiler pass '{name}' redefined")
                    compiler.passes[name] = _CompilerPass.from_toml(p)


@dataclass
class PreprocessorConfiguration:
    """
    Represents the configuration for a specific file, including:
    - Macro definitions
    - Include paths
    - Include files
    - A meaningful pass name
    """

    defines: list[str]
    include_paths: list[str]
    include_files: list[str]
    pass_name: str = "default"

    def _update(self, pass_or_mode: _CompilerPass | _CompilerMode):
        """
        Update this PreprocessorConfiguration by extending the
        defines, include paths and include files using the values
        contained in the provided _CompilerPass or _CompilerMode.

        Parameters
        ----------
        pass_or_mode: _CompilerPass | _CompilerMode
            The pass or mode to enable.
        """
        self.defines.extend(pass_or_mode.defines)
        self.include_paths.extend(pass_or_mode.include_paths)
        self.include_files.extend(pass_or_mode.include_files)


class ArgumentParser:
    """
    Represents the behavior of a specific compiler.
    """

    def __init__(self, path: str):
        self.name = os.path.basename(path)

        # Load the global compiler configuration if necessary.
        if not _compilers:
            _load_compilers()

        self.compiler = _Compiler()
        if self.name not in _compilers:
            log.warning(f"Compiler '{self.name}' not recognized.")
            return

        # If a compiler is not an alias, use its configuration directly.
        if not _compilers[self.name].alias_of:
            self.compiler = _compilers[self.name]
            log.info(f"Compiler '{self.name}' recognized.")

        # If a compiler is an alias of another, resolve the alias.
        # An alias may itself be an alias, so we may need to iterate.
        alias_chain = [self.name]
        while _compilers[alias_chain[-1]].alias_of:
            alias = _compilers[alias_chain[-1]].alias_of
            if alias in alias_chain:
                log.error(f"Compiler '{self.name}' alias results in a loop.")
                return
            if alias not in _compilers:
                log.error(
                    f"Compiler '{self.name}' aliases unrecognized '{alias}'.",
                )
                return
            alias_chain.append(alias)

        alias = alias_chain[-1]
        self.compiler = _compilers[alias]
        log.info(
            f"Compiler '{self.name}' recognized; aliases '{alias}'.",
        )

    def parse_args(self, argv: list[str]) -> list[PreprocessorConfiguration]:
        """
        Parameters
        ----------
        argv: list[str]
            The list of arguments passed to the compiler.

        Returns
        -------
        list[PreprocessorConfiguration]
            A list of compiler configurations, each representing
            a separate pass, that describe the compiler's behavior
            after parsing `argv`.
        """
        # There may be no actions defined with these destinations.
        namespace = argparse.Namespace()
        namespace.defines = []
        namespace.include_paths = []
        namespace.include_files = []
        namespace.modes = []

        # "passes" requires special handling to emulate compilers accurately.
        # - passes: Passes from built-in actions (e.g., store, append).
        # - _passes: Flag-specific defaults from custom actions.
        namespace.passes = []
        namespace._passes = {}

        # Configure the parser for arguments common to all compilers.
        parser = argparse.ArgumentParser(
            add_help=False,
            exit_on_error=False,
            allow_abbrev=False,
        )
        parser.add_argument("-D", dest="defines", action="append")
        parser.add_argument(
            "-I",
            "-isystem",
            dest="include_paths",
            action="append",
        )
        parser.add_argument(
            "-include",
            dest="include_files",
            action="append",
        )

        # Suppress warnings for common arguments we don't care about.
        parser.add_argument("-O", dest=None)
        parser.add_argument("-o", dest=None)
        parser.add_argument("-g", action="store_const", dest=None)
        parser.add_argument("-c", action="store_const", dest=None)
        parser.add_argument("file", nargs="*")

        # Add additional options for this specific compiler.
        for option in self.compiler.parser:
            kwargs = {k: v for k, v in option.items() if k != "flags"}

            # If a custom action, handle special-case for default passes.
            if not isinstance(kwargs["action"], str):
                if (
                    "dest" in kwargs
                    and kwargs["dest"] == "passes"
                    and "default" in kwargs
                ):
                    default_value = kwargs.pop("default")
                    flag_name = option["flags"][0]
                    namespace._passes[flag_name] = default_value
            parser.add_argument(*option["flags"], **kwargs)

        # Make a best-effort attempt to parse arguments.
        args, unrecognized = parser.parse_known_args(
            argv + self.compiler.options,
            namespace,
        )
        if unrecognized:
            log.warning(f"Unrecognized arguments: '{' '.join(unrecognized)}'")

        # Construct final list of active modes.
        args.modes = set(args.modes)

        # Construct final list of active passes.
        args.passes = set(args.passes)
        args.passes |= set(chain(*args._passes.values()))
        args.passes |= {"default"}

        # Convert the arguments into a list of preprocessor configurations.
        configurations = []
        for pass_name in args.passes:
            config = PreprocessorConfiguration(
                args.defines.copy(),
                args.include_paths.copy(),
                args.include_files.copy(),
                pass_name,
            )

            if pass_name == "default":
                modes = args.modes
            else:
                if pass_name not in self.compiler.passes:
                    log.error(f"Unrecognized compiler pass: {pass_name}")
                    continue
                config._update(self.compiler.passes[pass_name])
                modes = self.compiler.passes[pass_name].modes

            for mode_name in modes:
                if mode_name not in self.compiler.modes:
                    log.error(f"Unrecognized compiler mode: {mode_name}")
                    continue
                config._update(self.compiler.modes[mode_name])

            configurations.append(config)

        return configurations


def load_database(dbpath, rootdir):
    """
    Load a compilation database.
    Return a list of compilation commands, where each command is
    represented as a compilation database entry.
    """
    db = CompilationDatabase.from_file(dbpath)

    configuration = []
    for command in db:
        # Skip commands that invoke unsupported tools.
        if not command.is_supported():
            continue

        # Files may be specified:
        # - relative to root
        # - relative to a directory
        # - as an absolute path
        filedir = rootdir
        if command.directory is not None:
            if os.path.isabs(command.directory):
                filedir = command.directory
            else:
                filedir = os.path.abspath(
                    rootdir,
                    os.path.join(command.directory),
                )

        if os.path.isabs(command.filename):
            path = os.path.abspath(command.filename)
        else:
            path = os.path.abspath(os.path.join(filedir, command.filename))

        # Skip files that don't exist.
        # (e.g., because they're generated by running make)
        if not os.path.exists(path):
            log.warning(f"Ignoring non-existent file: {path}")
            continue

        # Parse command-line arguments, emulating compiler-specific behavior.
        compiler_name = os.path.basename(command.arguments[0])
        parser = ArgumentParser(compiler_name)
        preprocessor_configs = parser.parse_args(command.arguments[1:])

        # Create a configuration entry for each compiler pass.
        # Each compiler pass may set different defines, etc.
        for preprocessor_config in preprocessor_configs:
            entry = asdict(preprocessor_config)

            entry["file"] = path

            # Include paths may be specified relative to root
            entry["include_paths"] = [
                os.path.abspath(os.path.join(rootdir, f))
                for f in entry["include_paths"]
            ]

            configuration += [entry]

            # Print variables for debugging purposes.
            if not log.isEnabledFor(logging.DEBUG):
                continue
            pass_name = entry["pass_name"]
            for v in ["defines", "include_paths", "include_files"]:
                if entry[v]:
                    value = " ".join(entry[v])
                    log.debug(f"{v} for {path} in pass '{pass_name}': {value}")

    if len(configuration) == 0:
        log.warning(
            f"No files found in compilation database at '{dbpath}'.\n"
            + "Ensure that 'directory' and 'file' are in the root directory.",
        )

    return configuration
