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
import re

import pytest
import responses

from atldld.requests import RMAError, RMAParameters, rma


class TestRMAParameters:
    def test_model(self):
        params = RMAParameters("my-model")
        assert str(params) == "criteria=model::my-model"

    def test_criteria(self):
        criteria = {"id": 10, "name": "dataset"}
        params = RMAParameters("my-model", criteria=criteria)
        url_params = "criteria=model::my-model,rma::criteria,[id$eq10][name$eqdataset]"
        assert str(params) == url_params

    def test_include(self):
        params = RMAParameters("my-model", include=("genes", "section_images"))
        url_params = "criteria=model::my-model,rma::include,genes,section_images"
        assert str(params) == url_params

    @pytest.mark.parametrize(
        ("start_row", "num_rows", "expected_url_part"),
        (
            (5, 20, "[start_row$eq5][num_rows$eq20]"),
            (5, None, "[start_row$eq5]"),
            (None, 20, "[num_rows$eq20]"),
        ),
    )
    def test_options(self, start_row, num_rows, expected_url_part):
        params = RMAParameters("my-model", start_row=start_row, num_rows=num_rows)
        url_params = f"criteria=model::my-model,rma::options{expected_url_part}"
        assert str(params) == url_params


class TestRma:

    @responses.activate
    def test_request_works(self):
        params = RMAParameters("my-model")
        url_params = "criteria=model::my-model"
        msg = [1, 2, 3]
        status = {
            "success": True,
            "id": 0,
            "start_row": 0,
            "num_rows": len(msg),
            "total_rows": len(msg),
        }
        return_json = {**status, "msg": msg}

        responses.add(
            responses.GET,
            re.compile(fr"https://api.brain-map.org/api/.*/query.json\?{url_params}"),
            json=return_json,
        )
        status_got, msg_got = rma(params)
        assert status_got == status
        assert msg_got == msg

    @responses.activate
    def test_rma_error_raised(self):
        params = RMAParameters("my-model")
        msg = "some error occurred"
        return_json = {
            "success": False,
            "msg": msg,
        }
        responses.add(responses.GET, re.compile(""), json=return_json)
        with pytest.raises(RMAError, match=msg):
            rma(params)
