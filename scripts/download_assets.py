"""Fetch the third-party files we are not allowed to redistribute.

Two things: SPOCK's XGBoost JSON and their 1e9-orbit dataset. Both are
GPL-3.0 (or their own terms); see LICENSE. We download them instead of
putting them in the repo.

Safe to rerun: existing files are left alone. Downloads go to a temporary
name first, so a broken transfer does not leave a half file that looks
cached.
"""
import hashlib
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.normpath(os.path.join(HERE, "..", "models"))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

ASSETS = {
    os.path.join(MODELS, "featureclassifier.json"): (
        "https://raw.githubusercontent.com/dtamayo/spock/master/spock/models/featureclassifier.json",
        "ac0a7432e26377f5743ac88478d8cc65dbcdb18c57c9c166339c47478225c758",
    ),
    os.path.join(DATA, "spock_dataset.zip"): (
        "https://zenodo.org/api/records/14327471/files/dataset.zip/content",
        "7d458bab84992420d375ffd8bb605b7beff1e6c7c9851fe28bf868c7bb774c44",
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(path, url, expected):
    if os.path.exists(path):
        got = sha256(path)
        if got != expected:
            raise RuntimeError(
                f"{path} has the wrong hash.\n  expected {expected}\n  got      {got}\n"
                "Delete it and rerun if upstream changed.")
        print("cached", path)
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    print("downloading", url)
    with urllib.request.urlopen(url, timeout=60) as src, open(tmp, "wb") as dst:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
    got = sha256(tmp)
    if got != expected:
        os.remove(tmp)
        raise RuntimeError(
            f"{url} changed.\n  expected {expected}\n  got      {got}\n"
            "Update SHA256 in this script only after checking the new file.")
    os.replace(tmp, path)
    print("saved", path, os.path.getsize(path), "bytes")


if __name__ == "__main__":
    for path, (url, expected) in ASSETS.items():
        fetch(path, url, expected)
    print("done. Upstream credits are in README and LICENSE.")
