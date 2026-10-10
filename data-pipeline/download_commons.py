"""Download bill templates and real test photos from Wikimedia Commons, with license info."""
import json
import time

import requests

from config import COMMONS_REAL_TEST, COMMONS_TEMPLATES, RAW_TEMPLATES, REAL_TEST

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "krw-money-counter/0.1 (non-commercial portfolio project; python-requests)"}


def get(url, **kw):
    for attempt in range(8):
        r = requests.get(url, headers=HEADERS, timeout=60, **kw)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        wait = int(r.headers.get("Retry-After", 0)) or 15 * (attempt + 1)
        print(f"  rate limited, waiting {wait}s")
        time.sleep(wait)
    r.raise_for_status()


def file_info(title):
    r = get(API, params={
        "action": "query", "titles": f"File:{title}", "prop": "imageinfo",
        "iiprop": "url|extmetadata", "format": "json",
    })
    page = next(iter(r.json()["query"]["pages"].values()))
    info = page["imageinfo"][0]
    meta = info.get("extmetadata", {})
    return {
        "title": title,
        "url": info["url"],
        "page": info["descriptionurl"],
        "license": meta.get("LicenseShortName", {}).get("value"),
        "artist": meta.get("Artist", {}).get("value"),
    }


def download(titles, out_dir, prefix=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    credits = []
    for i, title in enumerate(titles):
        info = file_info(title)
        ext = title.rsplit(".", 1)[-1].lower()
        name = f"{prefix}{i:02d}.{ext}"
        path = out_dir / name
        if not path.exists():
            path.write_bytes(get(info["url"]).content)
        credits.append({**info, "file": name})
        print(f"{name}  <-  {title}  [{info['license']}]")
        time.sleep(3)
    return credits


def main():
    credits = []
    for cls, titles in COMMONS_TEMPLATES.items():
        credits += download(titles, RAW_TEMPLATES, prefix=f"{cls}_")
    (RAW_TEMPLATES / "CREDITS.json").write_text(json.dumps(credits, ensure_ascii=False, indent=2), "utf-8")

    real_credits = download(COMMONS_REAL_TEST, REAL_TEST / "images", prefix="commons_")
    (REAL_TEST / "CREDITS.json").write_text(json.dumps(real_credits, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()
