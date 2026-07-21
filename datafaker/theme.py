"""Entrypoint for the datafaker package."""
from dataclasses import dataclass
from enum import Enum

import colorama


colorama.just_fix_windows_console()


@dataclass
class Theme:
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
        prompt=colorama.Fore.CYAN + colorama.Style.NORMAL,
        column=colorama.Fore.GREEN + colorama.Style.NORMAL,
        data=colorama.Fore.YELLOW + colorama.Style.NORMAL,
        function=colorama.Fore.MAGENTA + colorama.Style.NORMAL,
        query=colorama.Fore.GREEN + colorama.Style.NORMAL,
        line=colorama.Fore.WHITE + colorama.Style.DIM,
        reset=colorama.Style.RESET_ALL,
    ),
    ThemeEntry.LIGHT: Theme(
        prompt=colorama.Fore.BLUE,
        column=colorama.Fore.GREEN,
        data=colorama.Fore.BLACK,
        function=colorama.Fore.MAGENTA,
        query=colorama.Fore.MAGENTA,
        line=colorama.Fore.LIGHTBLACK_EX,
        reset=colorama.Style.RESET_ALL,
    ),
}


theme_active = THEME[ThemeEntry.NONE]


def set_active_theme(te: ThemeEntry):
    global theme_active
    theme_active = THEME[te]


def get_active_theme() -> Theme:
    global theme_active
    return theme_active
