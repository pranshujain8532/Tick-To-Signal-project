"""Tests for the dashboard — the artefact most people will only ever see once.

WHY A PYTHON TEST SUITE FOR A JAVASCRIPT UI
    Because the properties worth guarding here are not behavioural, they are
    textual and structural, and those are exactly what a file-reading test can
    check without a browser, a bundler or a headless driver. The constitution
    forbids adding a JS toolchain, and none of these checks need one.

WHAT IS ACTUALLY BEING GUARDED
    CLAUDE.md forbids three claims "by construction, not by care". Care is a
    person remembering; construction is a test that fails. So:

      * the corrections that qualify the three forbidden claims must be present
        in the shipped markup, not injected at runtime, because a screenshot of
        a dashboard whose fetches failed is still a screenshot somebody will
        show to somebody else;
      * the numbers in that markup must still match the ones `serving/records.py`
        serves, or the static copy has rotted into a second, wrong source;
      * the palette is locked, so no tenth colour and no stray hex literal;
      * and there must be no build step, because "open the file" is the promise.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from serving import records

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = REPO_ROOT / "serving" / "dashboard"
INDEX = DASHBOARD / "index.html"
TOKENS = DASHBOARD / "css" / "tokens.css"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def collapsed(path: Path) -> str:
    """File text with runs of whitespace collapsed to single spaces.

    Prose in the markup wraps at 80 columns, so a sentence this suite is
    checking for is routinely split across two lines. Collapsing first means the
    test asserts on what a reader sees rather than on where the author pressed
    return — the same trick `test_serving.py` uses for the Stage 5 verdict.
    """
    return " ".join(read(path).split())


def javascript_files() -> list[Path]:
    return sorted(DASHBOARD.glob("js/**/*.js"))


def all_source_files() -> list[Path]:
    return sorted([INDEX, *DASHBOARD.glob("css/*.css"), *javascript_files()])


# ------------------------------------------------------- the three forbidden claims


def test_the_dashboard_ships_the_latency_correction_in_its_markup():
    """Forbidden claim 1: that low latency is what makes this signal tradeable.

    The correction names the measured half-life and the fee shortfall, and it
    is in `index.html` rather than only in the /latency payload — a correction
    that arrives by fetch is a correction that is absent exactly when the page
    is broken, which is when a misleading screenshot is most likely.
    """
    markup = collapsed(INDEX)

    assert "13.2 s" in markup, "the measured IC half-life must appear in the markup"
    assert "70" in markup, "the fee shortfall multiple must appear in the markup"
    assert "not what makes this signal tradeable" in markup

    # And the same two facts must still be the ones the server states, or the
    # static copy has drifted from the record it is quoting.
    assert "13.2 s" in records.RELEVANCE_NOTE
    assert "70x" in records.RELEVANCE_NOTE


def test_the_dashboard_never_calls_the_serving_figure_bare_latency():
    """Forbidden claim 2: that the microsecond figure is what is running here.

    The live readout is labelled with its runtime, so it cannot be read as the
    C++ number, and the Pareto panel states the measurement boundary and marks
    the two harnesses apart.
    """
    markup = collapsed(INDEX).lower()

    # The label on the live readout must name the runtime. Asserted on the words
    # rather than on the exact punctuation: the first version pinned the literal
    # string "serving (python / onnx int8)" and failed the day the separator
    # changed, which taught nothing — the property worth guarding is that the
    # number is never presented as an unqualified "latency".
    label = markup.split('id="serving-p50"')[0].rsplit('<span class="label">', 1)[-1]
    assert "serving" in label
    assert "python" in label and "onnx int8" in label
    assert not label.strip().startswith("latency")

    assert "forward pass only" in markup
    assert "python harness" in markup and "c++ harness" in markup
    assert "is not yet measured in c++" in markup


def test_the_pooled_ic_cannot_appear_without_its_correction():
    """Forbidden claim 3: that the pooled IC is the edge.

    The pooled value is not written into the markup at all — it is filled in
    from /stability at runtime — but the sentence that qualifies it IS in the
    markup, so the element can never be populated without the correction beside
    it. The per-block figure is the one the panel prints large.
    """
    markup = collapsed(INDEX)

    assert "inflated by common drift, not tradeable" in markup
    assert "per-block mean is what a trader would experience" in markup
    # The server refuses to serve it under a name that reads as an endorsement,
    # and the dashboard reads that key, so the guard holds end to end.
    assert "pooled_ic" not in records.stability_payload()
    assert "pooled_ic_not_tradeable" in read(DASHBOARD / "js" / "panels" / "stability.js")


def test_the_stability_panel_prints_the_per_block_figures_not_the_pooled_one():
    """If one number can be shown, it is 0.073 — so that is the one drawn large."""
    panel = read(DASHBOARD / "js" / "panels" / "stability.js")

    # The large numbers are the information ratio and the per-block mean.
    assert "data.information_ratio" in panel
    assert "signed(data.mean" in panel
    # The pooled value is written to a dimmed DOM element, never to the canvas.
    assert "pooledLabel.textContent" in panel
    assert "numberText" not in panel.split("function drawBlocks")[0].split("pooledLabel")[1]


# -------------------------------------------------------------- the locked palette


def test_the_palette_is_exactly_the_nine_locked_colours():
    """No tenth colour, and no drift in the nine.

    The design direction fixes these values. They are also read at runtime by
    every canvas panel through `readTokens`, so this file is the single source
    of truth for the whole dashboard and a change here changes everything.
    """
    expected = {
        "--ground": "#0B0F14",
        "--panel": "#121820",
        "--hairline": "#1F2933",
        "--text": "#E6EDF3",
        "--dim": "#7D8B9A",
        "--tape": "#E8A33D",
        "--signal": "#8FB5DA",
        "--bid": "#4F9E79",
        "--ask": "#C96A5F",
    }
    tokens = read(TOKENS)

    for name, value in expected.items():
        assert re.search(rf"{name}:\s*{value};", tokens), f"{name} must be {value}"

    declared = set(re.findall(r"(--[a-z-]+):\s*#", tokens))
    assert declared == set(expected), f"unexpected colour tokens: {declared - set(expected)}"


def test_no_colour_literal_appears_outside_the_token_file():
    """Canvas has no cascade, so a panel could trivially hardcode a colour.

    Every panel reads the palette through `canvas.js:readTokens` instead. This
    is the check that keeps that true: one stray `#4F9E79` in a draw call would
    survive any number of code reviews and would silently ignore a palette
    change.
    """
    offenders = []
    for path in all_source_files():
        if path == TOKENS:
            continue
        for number, line in enumerate(read(path).splitlines(), start=1):
            if re.search(r"#[0-9A-Fa-f]{3,8}\b", line) and "&#" not in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}")

    assert not offenders, "hard-coded colours outside tokens.css:\n" + "\n".join(offenders)


def test_rgb_alpha_variants_are_composed_from_the_same_nine_colours():
    """Alpha is the only permitted variation, and it reuses the locked values."""
    tokens = read(TOKENS)
    for name, value in [
        ("--tape-rgb", "232 163 61"),
        ("--signal-rgb", "143 181 218"),
        ("--bid-rgb", "79 158 121"),
        ("--ask-rgb", "201 106 95"),
    ]:
        assert f"{name}: {value};" in tokens


# ------------------------------------------------------------------ no build step


def test_there_is_no_build_step_and_no_external_asset():
    """`docker compose up` must serve this directly, offline, from a static mount.

    No npm, no bundler, no CDN. The container has no network and the promise is
    that the dashboard runs by hitting a route.
    """
    assert not (DASHBOARD / "package.json").exists()
    assert not (DASHBOARD / "node_modules").exists()

    for path in all_source_files():
        text = read(path)
        for pattern in ("http://", "https://", "cdn.", "unpkg", "jsdelivr"):
            assert pattern not in text, f"{path.name} reaches outside the container: {pattern}"


def test_every_asset_index_references_actually_exists():
    """A 404 on a module is a blank screen, and it is trivially preventable."""
    markup = read(INDEX)
    referenced = re.findall(r'(?:src|href)="([^"]+)"', markup)

    assert referenced, "index.html should reference its own stylesheets and modules"
    for reference in referenced:
        assert (DASHBOARD / reference).exists(), f"index.html references missing {reference}"


def test_every_module_is_imported_by_something():
    """No dead modules. The constitution bans dead code and this is where it hides."""
    imported = set()
    for path in javascript_files():
        for match in re.findall(r'from\s+"([^"]+)"', read(path)):
            imported.add(Path(match).name)
    imported.add("main.js")  # the entry point, referenced by index.html

    for path in javascript_files():
        assert path.name in imported, f"{path.name} is never imported"


# ---------------------------------------------------------- documentation discipline


@pytest.mark.parametrize("path", javascript_files(), ids=lambda p: p.name)
def test_every_module_opens_with_a_docstring_explaining_why(path: Path):
    """The constitution's first rule, applied to the JavaScript as well.

    Every module states WHAT it does and WHY it exists. The `why` check is
    deliberately crude — it looks for the word — because the alternative is not
    checking at all, and a file that cannot spare the word "why" has not
    explained itself.
    """
    text = read(path)

    assert text.startswith("/*"), f"{path.name} must open with a module docstring"
    header = text.split("*/", 1)[0].lower()
    assert "what" in header, f"{path.name} does not say what it does"
    assert "why" in header, f"{path.name} does not say why it exists"


def test_the_render_loop_is_the_only_animation_frame_request():
    """One loop, and panels that cannot start their own.

    A panel scheduling its own rAF is how a 4 ms budget becomes eight
    independent budgets that nobody measures. The boot sequence and the panels
    all draw from `render.js`; this asserts nothing else asks for a frame.
    """
    offenders = [
        path.name
        for path in javascript_files()
        # A CALL, not a mention: stream.js explains in prose why a hidden tab
        # stops acking, and that sentence is not a second render loop.
        if re.search(r"requestAnimationFrame\s*\(", read(path)) and path.name != "render.js"
    ]
    assert not offenders, f"only render.js may drive frames, but so does: {offenders}"


# ------------------------------------------------------------- the three pages


def test_the_dashboard_has_exactly_three_pages_and_a_tab_for_each():
    """The split that made the panels legible, guarded so it stays deliberate.

    Eight panels on one screen was unreadable at a real browser's ~890 usable
    pixels. Three pages of four is the fix, and a fourth page appearing without
    a tab — or a tab without a page — is the way that decays.
    """
    markup = read(INDEX)

    pages = set(re.findall(r'<main class="page" id="page-(\w+)"', markup))
    tabs = set(re.findall(r'<button class="tab" data-page="(\w+)"', markup))

    assert pages == {"live", "evidence", "system"}
    assert tabs == pages, "every page needs a tab and every tab needs a page"


def test_the_permanent_corrections_live_outside_every_page():
    """With more than one page, "beside the claim" has to mean "on every page".

    The header carries a microsecond figure on all three pages, so the sentence
    that stops it being read as the reason the signal works cannot live inside
    one of them. This asserts the relevance note is in the footer — which is
    chrome, outside `<main class="page">` — and would fail if somebody moved it
    into a panel.
    """
    markup = read(INDEX)

    footer = markup.split('<footer class="chrome"', 1)[1]
    assert "13.2 s" in footer
    assert "70" in footer
    assert "not what makes this signal tradeable" in footer

    # And it must not be inside any page, which is what would make it
    # conditional on which tab happens to be open.
    for page in re.findall(r'<main class="page".*?</main>', markup, flags=re.S):
        assert "not what makes this signal tradeable" not in page


def test_the_render_loop_skips_panels_whose_page_is_not_showing():
    """The page switch is a performance boundary, not only a navigational one.

    The depth tape is the expensive surface and it is on exactly one of the
    three pages; drawing it while somebody reads the Evidence page would spend
    the frame budget on pixels nobody is looking at.
    """
    loop = read(DASHBOARD / "js" / "render.js")
    assert "entry.page !== state.page" in loop
    assert "continue" in loop


def test_a_session_gap_never_dims_the_whole_screen():
    """The strobe, guarded against coming back.

    An earlier version faded every panel to 55% for 1.4 s on every session
    boundary. At 10x the committed set crosses one every ~7 s, so the screen
    flashed every few seconds and read as a rendering fault. A gap is already
    said three times where it happens — blank columns, a banner, a broken drift
    line — and the connection dim is reserved for a socket that is actually
    down.
    """
    css = read(DASHBOARD / "css" / "layout.css")
    state = read(DASHBOARD / "js" / "state.js")

    assert 'data-connection="gap"' not in css
    assert "gapDimUntil" not in state
    # The dim that remains is for a dropped socket, which is rare and important.
    assert 'data-connection="down"' in css
