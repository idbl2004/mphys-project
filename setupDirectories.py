"""Create the directories configured for MVTTools."""

from __future__ import annotations

import argparse
from pathlib import Path

from loadConfig import importConfiguration


REPOSITORY_ROOT = Path(__file__).resolve().parent
DIRECTORY_NAMES = (
    "dataPath",
    "processedDataPath",
    "sharedObjectsPath",
    "logPath",
)
DATA_SUBDIRECTORIES = ("fermiGBM", "GRBCatalogue")


def resolveConfiguredPath(configPath: Path, configuredPath: str) -> Path:
    """Resolve absolute paths as-is and relative paths from the config directory."""
    path = Path(configuredPath).expanduser()
    if path.is_absolute():
        return path
    return configPath.parent / path


def getDirectories(configFilePath: Path) -> list[Path]:
    """Return all directories required by the configured project layout."""
    config = importConfiguration(configFilePath)
    configuredDirectories = config.generalSettings.directories

    directories = [
        resolveConfiguredPath(
            configFilePath,
            str(getattr(configuredDirectories, directoryName)),
        )
        for directoryName in DIRECTORY_NAMES
    ]

    dataPath = resolveConfiguredPath(
        configFilePath,
        str(configuredDirectories.dataPath),
    )
    directories.extend(dataPath / subdirectory for subdirectory in DATA_SUBDIRECTORIES)
    return list(dict.fromkeys(directories))


def createDirectories(configFilePath: Path, dryRun: bool = False) -> list[Path]:
    """Create configured directories and return their resolved paths."""
    directories = getDirectories(configFilePath)
    for directory in directories:
        if not dryRun:
            directory.mkdir(parents=True, exist_ok=True)
    return directories


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the directory structure configured for MVTTools."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved directories without creating them.",
    )
    arguments = parser.parse_args()

    configFilePath = arguments.config.expanduser().resolve()
    for directory in createDirectories(configFilePath, dryRun=arguments.dry_run):
        action = "Would create" if arguments.dry_run else "Ensured"
        print(f"{action}: {directory}")


if __name__ == "__main__":
    main()