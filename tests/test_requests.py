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
import pytest

from atldld.requests import RMAParameters


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
        )
    )
    def test_options(self, start_row, num_rows, expected_url_part):
        params = RMAParameters("my-model", start_row=start_row, num_rows=num_rows)
        url_params = f"criteria=model::my-model,rma::options{expected_url_part}"
        assert str(params) == url_params
