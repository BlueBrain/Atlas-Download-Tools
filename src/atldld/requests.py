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
    """Abstraction of RMA parameters.

    For practical purposes not all RMA functionality is covered. Important
    restrictions are:
    * Criteria filters only support the equality operator
    * Associations (~nested criteria) are modeled by values in the `criteria`
      dictionary that are dictionaries themselves.
    * The include field does not support filters
    """

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
            # Associations are a way of specifying nested search filters and are
            # modeled by nested criteria, i.e. by values that are dictionaries
            # themselves.
            # For example, genes are an association of the section data set. So
            # to filter both on data set properties and on gene properties one
            # has specify something like this:
            # criteria = {"specimen_id": 123, "genes": {"acronym": "Gad1"}}
            # This should translate to the following URL part:
            # "rma::criteria,[specimen_id$eq123],genes[acronym$eqGad1]"
            # criteria = {"data_set":
            # {"specimen_id": 123, "genes": {"acronym": "Gad1"}}}
            # This should translate to the following URL part:
            # "rma::criteria,data_set[specimen_id$eq123](genes[acronym$eqGad1])"
            flags.append("rma::criteria")

            fields, associations = self._split_criteria(self.criteria)

            if fields:
                flags.append("".join(f"[{k}$eq{v}]" for k, v in fields.items()))

            for name, criteria in associations.items():
                # data_set, {"specimen_id": 123, "genes": {"acronym": "Gad1"}}
                flags.append(self._parse_association(name, criteria))

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

    def _split_criteria(
        self, criteria: Dict[str, Any]
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Separate criteria into fields and associations.

        For example: criteria: {"specimen_id": 123, "genes": {"acronym": "Gad1"}}
        --> fields: {"specimen_id": 123}
        --> associations: {"genes": {"acronym": "Gad1"}}
        """
        fields = {}
        associations = {}

        for k, v in criteria.items():
            if isinstance(v, dict):
                associations[k] = v
            else:
                fields[k] = v

        return fields, associations

    def _parse_association(self, name, criteria):
        """Parse association for the creation of the URL."""
        result = name
        fields, associations = self._split_criteria(criteria)

        for k, v in fields.items():
            result += f"[{k}$eq{v}]"

        if associations:
            parsed_associations = [
                self._parse_association(association_name, association_criteria)
                for association_name, association_criteria in associations.items()
            ]
            result += "("
            result += ",".join(parsed_associations)
            result += ")"

        return result


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
