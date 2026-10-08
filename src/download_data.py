"""Download Shakespeare's Complete Works (Project Gutenberg #100) to data/shakespeare.txt."""
import re
import urllib.request
from pathlib import Path

URLS = [
    "https://www.gutenberg.org/cache/epub/100/pg100.txt",
    "https://www.gutenberg.org/files/100/100-0.txt",
]
OUT = Path(__file__).resolve().parent.parent / "data" / "shakespeare.txt"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = None
    for url in URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8-sig", errors="ignore")
            print(f"Downloaded from {url}")
            break
        except Exception as exc:  # try next mirror
            print(f"Failed {url}: {exc}")
    if raw is None:
        raise SystemExit("Download failed. Save the text manually to data/shakespeare.txt")

    # Strip the Project Gutenberg header and license footer.
    start = re.search(r"\*\*\* START OF .*?\*\*\*", raw)
    end = re.search(r"\*\*\* END OF .*?\*\*\*", raw)
    if start:
        raw = raw[start.end():]
    if end:
        raw = raw.split(end.group(0))[0]
    OUT.write_text(raw.strip(), encoding="utf-8")
    print(f"Saved {len(raw):,} characters to {OUT}")


if __name__ == "__main__":
    main()
