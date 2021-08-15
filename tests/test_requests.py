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
