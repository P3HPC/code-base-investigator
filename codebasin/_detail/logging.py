#!/usr/bin/env python3
# Copyright (C) 2019-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import logging
import re


class Formatter(logging.Formatter):
    """
    A Formatter that formats LogRecords using a format inspired by compilers
    like gcc/clang, with optional colors.
    """

    def __init__(self, *, colors: bool = False):
        """
        Initialize this Formatter.

        Parameters
        ----------
        colors: bool, default: False
            Whether to colorize the output.
        """
        self.colors = colors

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the specified record.

        Parameters
        ----------
        record: logging.LogRecord
            The record to format.

        Returns
        -------
        str
            The formatted output string.
        """
        msg = record.msg
        level = record.levelname.lower()

        # Display info messages with no special formatting.
        if level == "info":
            return f"{msg}"

        # Drop colors if requested.
        if not self.colors:
            return f"{level}: {msg}"

        # Otherwise, use ASCII codes to improve readability.
        BOLD = "\033[1m"
        DEFAULT = "\033[39m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        RESET = "\033[0m"

        if level == "warning":
            color = YELLOW
        elif level == "error":
            color = RED
        else:
            color = DEFAULT

        # Print a stack trace if explicitly requested.
        if record.exc_info:
            return self.formatException(record.exc_info)

        return f"{BOLD}{color}{level}{RESET}: {msg}"


class MetaWarning:
    """
    A MetaWarning is used to represent multiple warnings, and to provide
    suggested actions to the user.
    """

    def __init__(self, regex: str, msg: str):
        """
        Initialize a new MetaWarning.

        Parameters
        ----------
        regex: str
            A regular expression used to identify constituent warnings.
            If any warning matches `regex`, this MetaWarning will trigger.

        msg: str
            The message to display when this MetaWarning is triggered.
        """
        self.regex = re.compile(regex)
        self.msg = msg
        self._count = 0

    def inspect(self, record: logging.LogRecord) -> bool:
        """
        Inspect a LogRecord to determine if it matches this MetaWarning.

        Parameters
        ----------
        record: logging.LogRecord
            The LogRecord to inspect.

        Returns
        -------
        bool
            True if `record` matches this MetaWarning and False otherwise.
        """
        if self.regex.search(record.msg):
            self._count += 1
            return True
        return False

    def warn(self, logger: logging.Logger) -> None:
        """
        Produce the warning associated with this MetaWarning.

        Parameters
        ----------
        log: logging.Logger
            The Logger that should be used to generate the MetaWarning.
        """
        if self._count == 0:
            return
        logger.warning(self.msg.format(self._count))


class WarningAggregator(logging.Filter):
    """
    A WarningAggregator is a logging.Filter that inspects warnings to generate
    meta-warnings and statistics. It does not perform any filtering, but uses
    the logging.Filter mechanism as a hook to automatically inspect every
    warning passing through a logger.
    """

    def __init__(self) -> None:
        self.meta_warnings = [
            MetaWarning(".", "{} warnings generated during preprocessing."),
            MetaWarning(
                "user include",
                "{} user include files could not be found.\n"
                + "  These could contain important macros and includes.\n"
                + "  Suggested solutions:\n"
                + "  - Check that the file(s) exist in the code base.\n"
                + "  - Check the include paths in the compilation database.\n"
                + "  - Check if the include(s) should have used '<>'.",
            ),
            MetaWarning(
                "system include",
                "{} system include files could not be found.\n"
                + "  These could define important feature macros.\n"
                + "  Suggested solutions:\n"
                + "  - Check that the file(s) exist on your system.\n"
                + "  - Use .cbi/config to define system include paths.\n"
                + "  - Use .cbi/config to define important macros.",
            ),
        ]

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Inspect the specified LogRecord, attempting to match it against each
        possible MetaWarning.

        Parameters
        ----------
        record: logging.LogRecord
            The record to inspect.

        Returns
        -------
        bool
            True, to prevent any warnings from being filtered.
        """
        if record.levelno == logging.WARNING:
            for meta_warning in self.meta_warnings:
                meta_warning.inspect(record)
        return True

    def warn(self, logger: logging.Logger) -> None:
        """
        Produce the warning associated with any MetaWarning(s) that were
        matched by this WarningAggregator.

        Parameters
        ----------
        logger: logging.Logger
            The Logger that should be used to generate the MetaWarning(s).
        """
        for meta_warning in self.meta_warnings:
            meta_warning.warn(logger)
