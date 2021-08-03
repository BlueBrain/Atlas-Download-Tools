"""Generate repository metadata."""
import datetime
import pathlib

import atldld

metadata_template = """\
---
packageurl: https://github.com/BlueBrain/Atlas-Download-Tools
major: {major_version}
minor: {minor_version}
description: Toolbox to download atlas data.
repository: https://github.com/BlueBrain/Atlas-Download-Tools
externaldoc: https://bbpteam.epfl.ch/documentation/a.html#Atlas-Download-Tools
updated: {date}
maintainers: Francesco Casalegno
name: Atlas Download Tools
license: BBP-internal-confidential
issuesurl: https://github.com/BlueBrain/Atlas-Download-Tools/issues
version: {version}
contributors: Francesco Casalegno
---
"""

file_directory = pathlib.Path(__file__).parent.resolve()
metadata_path = file_directory / "metadata.md"

version = atldld.__version__
major_version = version.split(".")[0]
minor_version = version.split(".")[1]
date = datetime.datetime.now().strftime("%d/%m/%y")

metadata_instance = metadata_template.format(
    version=version, major_version=major_version, minor_version=minor_version, date=date
)

with metadata_path.open("w") as f:
    f.write(metadata_instance)

print("Finished")
