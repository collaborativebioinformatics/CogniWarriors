#!/usr/bin/env python3
"""Download small (non-imaging) files needed for the phenotype pipeline from
the public OpenNeuro S3 bucket for ds007116 ("Penn LEAD").

The bucket is public and browsable via plain HTTPS GET to the S3 REST API,
no auth needed. Three jobs, each independently selectable:

  --participants    ds007116/participants.tsv + .json (root, 2 known keys)
  --sessions        ds007116/sub-<ID>/sub-<ID>_sessions.{tsv,json} for all
                    132 subjects (subject IDs discovered via one delimited
                    listing call, then each file fetched directly -- no need
                    to list inside every subject folder)

Structural-MRI derivatives (FreeSurfer stats etc.) are explicitly out of
scope for this downloader -- that's the image-side teammate's territory.

All downloads are idempotent (skipped if the local file already matches the
remote Content-Length) and atomic (written to a .part file, then renamed).
"""

import argparse
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import config

S3_BASE = "https://s3.amazonaws.com/openneuro.org"
DATASET = "ds007116"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

SUBJECT_PREFIX_RE = re.compile(rf"^{DATASET}/(sub-\d+)/$")


def list_delimited(prefix):
    """One non-recursive listing call: returns (contents, common_prefixes)."""
    url = (
        f"{S3_BASE}/?list-type=2&delimiter=/"
        f"&prefix={urllib.parse.quote(prefix)}"
    )
    with urllib.request.urlopen(url) as resp:
        root = ET.fromstring(resp.read())
    contents = [
        (c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text))
        for c in root.findall("s3:Contents", NS)
    ]
    common_prefixes = [
        cp.find("s3:Prefix", NS).text for cp in root.findall("s3:CommonPrefixes", NS)
    ]
    return contents, common_prefixes


def head_size(key):
    url = f"{S3_BASE}/{urllib.parse.quote(key)}"
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as resp:
        return int(resp.headers.get("Content-Length", -1))


def download(key, dest_path, expected_size=None, retries=3):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{S3_BASE}/{urllib.parse.quote(key)}"
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            size = expected_size if expected_size is not None else head_size(key)
            if dest_path.exists() and dest_path.stat().st_size == size:
                return "skip"
            urllib.request.urlretrieve(url, part_path)
            part_path.replace(dest_path)
            return "downloaded"
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                return "missing"
            if attempt == retries:
                return "failed"
            time.sleep(1.5 * attempt)
    return "failed"


def run_job(name, jobs):
    print(f"\n=== {name} ===")
    counts = {"downloaded": 0, "skip": 0, "missing": 0, "failed": 0}
    for key, dest_path, size in jobs:
        status = download(key, dest_path, expected_size=size)
        counts[status] = counts.get(status, 0) + 1
        if status in ("failed", "missing"):
            print(f"  {status}: {key}")
    print(f"  downloaded={counts['downloaded']} skip={counts['skip']} "
          f"missing={counts['missing']} failed={counts['failed']}")
    return counts


def job_participants():
    for fname in ("participants.tsv", "participants.json"):
        key = f"{DATASET}/{fname}"
        yield key, config.DATA_ROOT / fname, None


def job_sessions():
    _, common_prefixes = list_delimited(f"{DATASET}/")
    subject_ids = sorted(
        m.group(1) for p in common_prefixes if (m := SUBJECT_PREFIX_RE.match(p))
    )
    print(f"  discovered {len(subject_ids)} subjects")
    for sub in subject_ids:
        for ext in ("tsv", "json"):
            key = f"{DATASET}/{sub}/{sub}_sessions.{ext}"
            yield key, config.SESSIONS_DIR / f"{sub}_sessions.{ext}", None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", action="store_true")
    parser.add_argument("--sessions", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not any([args.participants, args.sessions, args.all]):
        parser.error("pass at least one of --participants --sessions --all")

    if args.participants or args.all:
        run_job("participants.tsv/.json", job_participants())
    if args.sessions or args.all:
        run_job("sub-*_sessions.{tsv,json}", job_sessions())


if __name__ == "__main__":
    main()
