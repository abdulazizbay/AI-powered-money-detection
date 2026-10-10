"""Download the DTD texture dataset and keep only table/cloth-like categories as backgrounds."""
import tarfile

import requests
from tqdm import tqdm

from config import BACKGROUNDS, DATA, DTD_CATEGORIES, DTD_URL


def main():
    archive = DATA / "raw" / "dtd.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        part = archive.with_suffix(".part")
        # resume partial downloads (the server is slow and drops connections)
        for _ in range(20):
            done = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={done}-"} if done else {}
            try:
                with requests.get(DTD_URL, stream=True, timeout=60, headers=headers) as r:
                    if r.status_code == 416:
                        break  # already complete
                    r.raise_for_status()
                    total = done + int(r.headers.get("content-length", 0))
                    mode = "ab" if r.status_code == 206 else "wb"
                    with open(part, mode) as f, tqdm(total=total, initial=done if mode == "ab" else 0,
                                                     unit="B", unit_scale=True) as bar:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                            bar.update(len(chunk))
                break
            except requests.exceptions.RequestException as e:
                print(f"connection dropped ({e.__class__.__name__}), resuming...")
        part.rename(archive)

    BACKGROUNDS.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive) as tar:
        for m in tar:
            parts = m.name.split("/")  # dtd/images/<category>/<file>.jpg
            if m.isfile() and len(parts) == 4 and parts[1] == "images" and parts[2] in DTD_CATEGORIES:
                data = tar.extractfile(m).read()
                (BACKGROUNDS / f"dtd_{parts[3]}").write_bytes(data)
                count += 1
    print(f"{count} backgrounds -> {BACKGROUNDS}")


if __name__ == "__main__":
    main()
