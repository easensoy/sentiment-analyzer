"""Screenshot FastAPI's /docs page answering example sentences.

Start the API, then run from the repo root:

    python -m uvicorn src.api:app --port 8000
    python scripts/docs_screenshot.py --url http://127.0.0.1:8000

Drives the locally installed Microsoft Edge (or Chrome) headless through Playwright,
so no browser download is needed. For each sentence it fills POST /predict in
Swagger UI, presses Execute and captures the request body down to the server's
response. Writes:

    visual_outputs/05_fastapi_docs_examples.png   the four answers on one sheet
    visual_outputs/05a_fastapi_docs_page.png      one unedited capture of the page

Needs playwright and pillow (dev-only, not in requirements.txt).
"""
import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "visual_outputs"
OP = "#operations-default-predict_sentiment_predict_post"
LABELS = {0: "negative", 1: "positive"}
EXAMPLES = [
    ("positive", "An absolute masterpiece: superb acting and a story I loved from start to finish."),
    ("negative", "Boring, predictable and badly acted. A total waste of two hours."),
    ("positive", "I expected to hate it, but it turned out to be surprisingly good."),
    ("negative", "The movie was not good."),
]
SURFACE, INK, INK_2, HAIRLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"


def launch(p):
    error = None
    for channel in ("msedge", "chrome"):
        try:
            return p.chromium.launch(channel=channel, headless=True)
        except Exception as exc:  # channel not installed: try the next one
            error = exc
    raise SystemExit(f"no installed Edge or Chrome for Playwright: {error}")


def capture(url, tmp):
    shots = []
    with sync_playwright() as p:
        browser = launch(p)
        page = browser.new_page(viewport={"width": 960, "height": 2400}, device_scale_factor=2)
        page.goto(f"{url}/docs")
        page.click(f"{OP} .opblock-summary")
        page.click(f"{OP} button.try-out__btn")
        for i, (expected, text) in enumerate(EXAMPLES, 1):
            clear = page.locator(f"{OP} button.btn-clear")
            if clear.count():
                clear.click()
            page.fill(f"{OP} textarea.body-param__text", json.dumps({"text": text}, indent=2))
            page.click(f"{OP} button.execute")
            body = page.locator(f"{OP} .live-responses-table tbody pre.microlight").first
            body.wait_for()
            status = page.locator(f"{OP} .live-responses-table tbody .response-col_status").first.inner_text().strip()
            answer = json.loads(body.inner_text())
            page.evaluate("window.scrollTo(0, 0)")  # the tall viewport holds the whole operation
            op = page.locator(OP).bounding_box()
            top = page.locator(f"{OP} .opblock-section-request-body").bounding_box()["y"] - 8
            end = body.bounding_box()
            bottom = end["y"] + end["height"] + 4  # stop above the "Response headers" label
            path = Path(tmp) / f"example_{i}.png"
            page.screenshot(path=path, clip={"x": op["x"], "y": top, "width": op["width"], "height": bottom - top})
            shots.append({"n": i, "expected": expected, "text": text, "status": status,
                          "answer": answer, "image": Image.open(path).copy()})
            print(f"{i}. HTTP {status} {answer} <- {text}")
            if i == 1:
                page.screenshot(path=OUT / "05a_fastapi_docs_page.png",
                                clip={"x": 0, "y": 0, "width": 960, "height": bottom})
        browser.close()
    return shots


def font(name, size):
    try:
        return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)
    except OSError:
        return ImageFont.load_default(size)


def compose(shots):
    tile_w, gap, margin = 1100, 36, 56
    title, caption = font("seguisb.ttf", 40), font("segoeui.ttf", 27)
    subtitle = font("segoeui.ttf", 27)
    tiles = []
    for s in shots:
        img = s["image"]
        img = img.resize((tile_w, round(img.height * tile_w / img.width)), Image.LANCZOS)
        got = s["answer"].get("sentiment")
        verdict = "correct" if LABELS.get(got) == s["expected"] else "wrong"
        tiles.append((f"{s['n']}. Expected {s['expected']}; API returned {got} ({LABELS.get(got)}), {verdict}", img))
    cols = 2
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    cap_h = 52
    row_heights = [cap_h + max(img.height for _, img in row) for row in rows]
    width = margin * 2 + cols * tile_w + (cols - 1) * gap
    head_h = 150
    height = margin + head_h + sum(row_heights) + gap * (len(rows) - 1) + margin
    sheet = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, margin), "FastAPI /docs answering four example reviews", font=title, fill=INK)
    draw.text((margin, margin + 64), "POST /predict in Swagger UI, served by uvicorn from src.api:app; each answer "
              "gives the label, the 0/1 sentiment and the SVM margin (a distance, not a probability)",
              font=subtitle, fill=INK_2)
    y = margin + head_h
    for row, row_h in zip(rows, row_heights):
        x = margin
        for text, img in row:
            draw.text((x, y), text, font=caption, fill=INK)
            sheet.paste(img, (x, y + cap_h))
            draw.rectangle([x - 1, y + cap_h - 1, x + img.width, y + cap_h + img.height], outline=HAIRLINE, width=2)
            x += tile_w + gap
        y += row_h + gap
    path = OUT / "05_fastapi_docs_examples.png"
    sheet.save(path)
    print(f"wrote {path.relative_to(ROOT)} {sheet.size} and visual_outputs/05a_fastapi_docs_page.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Screenshot /docs answering example sentences")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="where the API is running")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        compose(capture(args.url, tmp))


if __name__ == "__main__":
    main()
