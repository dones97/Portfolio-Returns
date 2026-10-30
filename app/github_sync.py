"""Commit ticker mappings to GitHub, as Streamlit Cloud's disk is temporary."""
import base64
import csv
import io

import requests

GITHUB_API_URL = "https://api.github.com"
# A second attempt covers another commit landing between the read and the write.
MAX_COMMIT_ATTEMPTS = 2


def _mapping_line(code, ticker):
    """Format one CSV line, quoting a company name that contains a comma."""
    out = io.StringIO()
    csv.writer(out, lineterminator="").writerow([code, ticker])
    return out.getvalue()


def merge_mappings_csv(existing_text, new_mappings):
    """Return the CSV text with new mappings applied, keeping every other line."""
    line_ending = "\r\n" if "\r\n" in existing_text else "\n"
    remaining = dict(new_mappings)
    lines = []
    for line in existing_text.splitlines():
        row = next(csv.reader([line]), [])
        code = row[0].strip() if row else ""
        if code in remaining:
            lines.append(_mapping_line(code, remaining.pop(code)))
        else:
            lines.append(line)
    lines.extend(_mapping_line(code, t) for code, t in remaining.items())
    return line_ending.join(lines) + line_ending


def _raise_for_github_error(resp):
    """Raise on a failed response, keeping the reason GitHub gives in its body."""
    if resp.ok:
        return
    try:
        reason = resp.json()["message"]
    except (ValueError, KeyError, TypeError):
        reason = resp.text[:300]
    raise ValueError(f"GitHub returned HTTP {resp.status_code}. {reason}")


def save_mappings_to_github(token, repo, branch, path, new_mappings):
    """Merge mappings into the file on GitHub and commit, re-reading it first.

    Reading the latest version means mappings edited on GitHub are never lost.
    """
    url = f"{GITHUB_API_URL}/repos/{repo}/contents/{path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
    }
    message = "Add ticker mappings: " + ", ".join(sorted(new_mappings))

    for attempt in range(MAX_COMMIT_ATTEMPTS):
        resp = requests.get(
            url, headers=headers, params={"ref": branch}, timeout=30
        )
        _raise_for_github_error(resp)
        current = resp.json()
        text = base64.b64decode(current["content"]).decode("utf-8")
        payload = {
            "message": message,
            "content": base64.b64encode(
                merge_mappings_csv(text, new_mappings).encode("utf-8")
            ).decode("ascii"),
            "sha": current["sha"],
            "branch": branch,
        }
        resp = requests.put(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 409 and attempt + 1 < MAX_COMMIT_ATTEMPTS:
            continue
        _raise_for_github_error(resp)
        return
