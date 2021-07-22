"""The command line interface (CLI) for Atlas-Download-Tools."""
import click

import atldld


@click.group(
    help="""\
    Atlas-Download-Tools command line interface.

    For detailed instructions see the documentation of the corresponding sub-commands.
    """
)
def root():
    """Run the command line interface for Atlas-Download-Tools."""


@click.group(help="Informational subcommands.")
def info():
    """Run informational subcommands."""


@click.command(help="Version of Atlas-Download-Tools.")
def version():
    """Print the version of Atlas-Download-Tools."""
    click.echo(f"Atlas-Download-Tools version {atldld.__version__}")


@click.command(help="Location of the global cache folder.")
def cache_folder():
    """Print the location of the global cache folder."""
    # Slow import
    # TODO: move this constant somewhere else where the import is fast.
    from atldld.base import GLOBAL_CACHE_FOLDER

    click.echo("Location of the global cache folder:")
    click.echo(str(GLOBAL_CACHE_FOLDER.resolve()))


root.add_command(info)
info.add_command(version)
info.add_command(cache_folder)

if __name__ == "__main__":
    root()
