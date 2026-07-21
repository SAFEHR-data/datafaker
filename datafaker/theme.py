"""Entrypoint for the datafaker package."""
from dataclasses import dataclass
from enum import Enum

import colorama

colorama.just_fix_windows_console()


@dataclass
class Theme:
    """A colour theme for DataFaker terminal output."""

    prompt: str
    column: str
    data: str
    function: str
    query: str
    line: str
    reset: str


class ThemeEntry(str, Enum):
    """Themes available in the ``--theme`` option."""

    NONE = "none"
    DARK = "dark"
    LIGHT = "light"


THEME: dict[str, Theme] = {
    ThemeEntry.NONE: Theme("", "", "", "", "", "", ""),
    ThemeEntry.DARK: Theme(
        prompt=colorama.Fore.CYAN + colorama.Style.NORMAL,  # type: ignore
        column=colorama.Fore.GREEN + colorama.Style.NORMAL,  # type: ignore
        data=colorama.Fore.YELLOW + colorama.Style.NORMAL,  # type: ignore
        function=colorama.Fore.MAGENTA + colorama.Style.NORMAL,  # type: ignore
        query=colorama.Fore.GREEN + colorama.Style.NORMAL,  # type: ignore
        line=colorama.Fore.WHITE + colorama.Style.DIM,  # type: ignore
        reset=colorama.Style.RESET_ALL,  # type: ignore
    ),
    ThemeEntry.LIGHT: Theme(
        prompt=colorama.Fore.BLUE,  # type: ignore
        column=colorama.Fore.GREEN,  # type: ignore
        data=colorama.Fore.BLACK,  # type: ignore
        function=colorama.Fore.MAGENTA,  # type: ignore
        query=colorama.Fore.MAGENTA,  # type: ignore
        line=colorama.Fore.LIGHTBLACK_EX,  # type: ignore
        reset=colorama.Style.RESET_ALL,  # type: ignore
    ),
}


theme_active = THEME[ThemeEntry.NONE]


def set_active_theme(te: ThemeEntry):
    """Set the active theme by key."""
    global theme_active  # pylint: disable=global-statement
    theme_active = THEME[te]


def get_active_theme() -> Theme:
    """Get the active theme."""
    return theme_active
