# The package atldld is a tool to download atlas data.
#
# Copyright (C) 2021 EPFL/Blue Brain Project
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""REST API requests to the AIBS servers.

More info: https://help.brain-map.org/display/api/Allen+Brain+Atlas+API
"""
import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)

_API_BASE_URL = "https://api.brain-map.org/api/v2"


class RMAError(Exception):
    """Raised when the RMA response status is False."""


@dataclass
class RMAParameters:
    """Abstraction of RMA parameters."""

    model: str
    criteria: Optional[Dict[str, Any]] = None
    include: Optional[Sequence[str]] = None
    start_row: Optional[int] = None
    num_rows: Optional[int] = None

    def __str__(self) -> str:
        """Convert RMA parameters to URL parameters."""
        flags = [f"model::{self.model}"]

        # Criteria
        if self.criteria is not None:
            flags.append("rma::criteria")
            flags.append("".join(f"[{k}$eq{v}]" for k, v in self.criteria.items()))

        # Include
        if self.include is not None:
            flags.append("rma::include")
            flags.extend(self.include)

        # Options
        if self.start_row is not None or self.num_rows is not None:
            flag = "rma::options"
            if self.start_row is not None:
                flag += f"[start_row$eq{self.start_row}]"
            if self.num_rows is not None:
                flag += f"[num_rows$eq{self.num_rows}]"
            flags.append(flag)

        return f'criteria={",".join(flags)}'


def rma_all(rma_parameters: RMAParameters) -> list:
    """Send one or multiple RMA requests to get all data for given parameters.

    Parameters
    ----------
    rma_parameters
        The RMA query parameters. Corresponds to the part of the query URL
        that follows the "?" character. The fields start_row and num_rows
        must be equal to None because they are used/changed in this function

    Returns
    -------
    msg : list
        The data received from the server.
    """
    # Make a copy of the parameters (since we'll modify them) and at the same
    # time set start_row and num_row. The value of 25k rows per request is
    # the max on the AIBS side.
    rma_parameters = replace(rma_parameters, start_row=0, num_rows=25000)

    # Initial request. Chances are we already get all the data from it
    status, msg = rma(rma_parameters)
    start_row = status["start_row"]
    num_rows = status["num_rows"]
    total_rows = status["total_rows"]

    # We didn't specify any value for start_row in the initial request above, so
    # it should be set to 0 by default.
    if start_row != 0:
        raise RuntimeError(f"Expected start_row to be 0 but got {start_row}")

    # If not all data was received on the initial request, then send further
    # requests to get the remaining items
    pos = num_rows
    while pos < total_rows:
        rma_parameters.start_row = pos
        status, new_msg = rma(rma_parameters)

        # Check if the reported total_rows is consistent with initial response
        if status["total_rows"] != total_rows:
            raise RuntimeError(
                f'Expected total_rows to be {total_rows} but got {status["total_rows"]}'
            )

        # Each new request should yield new data. If no data was received, then
        # something must have gone wrong.
        if status["num_rows"] == 0:
            raise RuntimeError("No data received")

        pos += status["num_rows"]
        msg += new_msg

    return msg


def rma(rma_parameters: RMAParameters) -> Tuple[dict, list]:
    """Send a single RMA query and separate status and data from the response.

    Parameters
    ----------
    rma_parameters
        The RMA query parameters. Corresponds to the part of the query URL
        that follows the "?" character.

    Returns
    -------
    status : dict
        The response status. Has keys "success", "id", "start_row", "num_rows",
        "total_rows".
    msg : list
        The requested data.
    """
    url = f"{_API_BASE_URL}/data/query.json?{rma_parameters}"
    response = requests.get(url)
    response.raise_for_status()

    status = response.json()
    msg = status.pop("msg")

    # If success = False then msg is a string with the error description
    if not status["success"]:
        raise RMAError(f"{msg}\nURL: {url}")

    logger.debug("Total rows: %d", status["total_rows"])

    return status, msg
