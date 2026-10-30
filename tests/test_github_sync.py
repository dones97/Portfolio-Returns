"""Tests for committing ticker mappings to GitHub, using no network access."""
import base64
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import github_sync  # noqa: E402

ARGS = ("token", "owner/repo", "main", "ticker_mappings.csv")


def response(status, data=None):
    """Build a fake GitHub response."""
    resp = mock.Mock(ok=status < 400, status_code=status, text=str(data))
    resp.json.return_value = data
    return resp


def file_response(text, sha="sha1"):
    """Build a fake contents response, wrapped in lines as GitHub's base64 is."""
    encoded = base64.encodebytes(text.encode("utf-8")).decode("ascii")
    return response(200, {"content": encoded, "sha": sha})


def committed_text(put):
    """Decode the file content sent in a mocked PUT call."""
    return base64.b64decode(put.call_args.kwargs["json"]["content"]).decode("utf-8")


def test_merge_appends_new_mapping_and_keeps_other_lines_unchanged():
    text = "scrip_code,yahoo_ticker\r\n500049,BEL.NS\r\nMCX LTD, MCX.NS\r\n"
    assert github_sync.merge_mappings_csv(text, {"KRN": "KRN.NS"}) == (
        "scrip_code,yahoo_ticker\r\n500049,BEL.NS\r\nMCX LTD, MCX.NS\r\n"
        "KRN,KRN.NS\r\n"
    )


def test_merge_replaces_an_existing_code_in_place():
    text = "scrip_code,yahoo_ticker\n500049,OLD.NS\n500050,X.NS\n"
    assert github_sync.merge_mappings_csv(text, {"500049": "BEL.NS"}) == (
        "scrip_code,yahoo_ticker\n500049,BEL.NS\n500050,X.NS\n"
    )


def test_merge_quotes_a_company_name_containing_a_comma():
    merged = github_sync.merge_mappings_csv("scrip_code,yahoo_ticker\n", {"A, B": "C.NS"})
    assert merged == 'scrip_code,yahoo_ticker\n"A, B",C.NS\n'
    again = github_sync.merge_mappings_csv(merged, {"A, B": "D.NS"})
    assert again == 'scrip_code,yahoo_ticker\n"A, B",D.NS\n'


def test_commit_sends_merged_file_with_the_sha_it_read():
    existing = "scrip_code,yahoo_ticker\n500049,BEL.NS\n"
    with mock.patch.object(github_sync.requests, "get") as get, mock.patch.object(
        github_sync.requests, "put"
    ) as put:
        get.return_value = file_response(existing)
        put.return_value = response(200, {})
        github_sync.save_mappings_to_github(*ARGS, {"KRN": "KRN.NS"})
    body = put.call_args.kwargs["json"]
    assert body["sha"] == "sha1"
    assert body["branch"] == "main"
    assert body["message"] == "Add ticker mappings: KRN"
    assert committed_text(put) == existing + "KRN,KRN.NS\n"
    assert put.call_args.kwargs["headers"]["Authorization"] == "Bearer token"


def test_commit_rereads_the_file_when_another_commit_got_in_first():
    with mock.patch.object(github_sync.requests, "get") as get, mock.patch.object(
        github_sync.requests, "put"
    ) as put:
        get.side_effect = [
            file_response("scrip_code,yahoo_ticker\n", "old"),
            file_response("scrip_code,yahoo_ticker\nNEW,NEW.NS\n", "new"),
        ]
        put.side_effect = [response(409, {"message": "conflict"}), response(200, {})]
        github_sync.save_mappings_to_github(*ARGS, {"KRN": "KRN.NS"})
    assert put.call_args.kwargs["json"]["sha"] == "new"
    assert "NEW,NEW.NS" in committed_text(put)


def test_commit_failure_reports_the_reason_from_github():
    with mock.patch.object(github_sync.requests, "get") as get, mock.patch.object(
        github_sync.requests, "put"
    ) as put:
        get.return_value = file_response("scrip_code,yahoo_ticker\n")
        put.return_value = response(403, {"message": "Resource not accessible"})
        with pytest.raises(ValueError, match="HTTP 403. Resource not accessible"):
            github_sync.save_mappings_to_github(*ARGS, {"KRN": "KRN.NS"})
