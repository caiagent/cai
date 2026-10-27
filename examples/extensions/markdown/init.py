"""init.py - example extension: render the assistant's markdown in the TUI.

a cai.renderer for the `response` kind: the transcript hands it the whole
assistant message so far (ctx.text) on every streamed chunk and once more when
the message ends (ctx.done), and paints what it returns in place of the raw
markdown. the pass is line-oriented and stdlib-only, so it is fast enough to
run per chunk and renders an unfinished message as well as a finished one -
an open code fence is still a code block, an unmatched `**` stays literal.

what it renders: headings (bold, # stripped), bullet and task lists (•, ☐, ☑),
numbered lists (kept), block quotes (a bar), horizontal rules, fenced code
(a framed block, the language labelled), pipe tables (columns aligned), and
inline **bold**, *italic*, ~~strike~~, `code` and [links](url).

every styled span re-opens ctx.style (the block's base color) after its reset,
so the text around it keeps the transcript's assistant color. width is a
hint: tables are laid out against it, everything else wraps on the screen."""
import re

import cai


_ESC = "\033["
_RESET = _ESC + "m"
_BOLD = _ESC + "1m"
_DIM = _ESC + "2m"
_ITALIC = _ESC + "3m"
_UNDERLINE = _ESC + "4m"
_STRIKE = _ESC + "9m"
_CODE = _ESC + "38;5;215m"       # inline code and fenced code text
_LINK = _ESC + "38;5;75m"        # link text

_FENCE = re.compile(r"^(\s*)(```+|~~~+)\s*([\w.+-]*)\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_QUOTE = re.compile(r"^(\s*)>\s?(.*)$")
_BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_TASK = re.compile(r"^\[([ xX])\]\s+(.*)$")
_NUMBERED = re.compile(r"^(\s*)(\d+[.)])\s+(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")

_CODE_SPAN = re.compile(r"(`+)(.+?)\1")
_BOLD_SPAN = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC_SPAN = re.compile(r"(?<![\w*])(\*|_)(?=\S)(.+?)(?<=\S)\1(?![\w*])")
_STRIKE_SPAN = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
_LINK_SPAN = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def _span(style, text, base):
    """one styled span, closing back to the block's base style."""
    return style + text + _RESET + base


def _inline(text, base):
    """the inline markdown of one line: code spans first (their contents are
    left alone), then bold / italic / strike / links in what is left."""
    out = []
    pos = 0
    for m in _CODE_SPAN.finditer(text):
        out.append(_emphasis(text[pos:m.start()], base))
        out.append(_span(_CODE, m.group(2), base))
        pos = m.end()
    out.append(_emphasis(text[pos:], base))
    return "".join(out)


def _emphasis(text, base):
    if not text:
        return text
    text = _LINK_SPAN.sub(lambda m: _span(_LINK + _UNDERLINE, m.group(1), base)
                          + _span(_DIM, f" ({m.group(2)})", base), text)
    text = _BOLD_SPAN.sub(lambda m: _span(_BOLD, m.group(2), base), text)
    text = _ITALIC_SPAN.sub(lambda m: _span(_ITALIC, m.group(2), base), text)
    text = _STRIKE_SPAN.sub(lambda m: _span(_STRIKE, m.group(1), base), text)
    return text


def _cells(row):
    """the cells of one pipe-table row, outer pipes dropped, stripped."""
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [cell.strip() for cell in row.split("|")]


def _table(rows, base):
    """a pipe table laid out in aligned columns: the header bold, the rule
    row a line, cells joined by a dim bar. rows are the raw lines."""
    grid = []
    rule_at = None
    for i, row in enumerate(rows):
        if _TABLE_RULE.match(row) and rule_at is None and i > 0:
            rule_at = i
            grid.append(None)
            continue
        grid.append(_cells(row))
    columns = 0
    for cells in grid:
        if cells is not None:
            columns = max(columns, len(cells))
    widths = [0] * columns
    for cells in grid:
        if cells is None: continue
        for j, cell in enumerate(cells):
            widths[j] = max(widths[j], len(cell))
    out = []
    bar = _span(_DIM, " │ ", base)
    for i, cells in enumerate(grid):
        if cells is None:
            out.append(_span(_DIM, "─┼─".join("─" * w for w in widths), base))
            continue
        padded = []
        for j in range(columns):
            cell = cells[j] if j < len(cells) else ""
            pad = " " * (widths[j] - len(cell))
            text = _inline(cell, base)
            if rule_at is not None and i < rule_at:
                text = _span(_BOLD, cell, base)
            padded.append(text + pad)
        out.append(bar.join(padded).rstrip())
    return out


def render_markdown(text, width=80, base=""):
    """the block-level pass: fences, tables, headings, rules, quotes, lists,
    then the inline pass on everything else. returns the rendered text."""
    out = []
    in_fence = None          # the fence marker while inside a code block
    table = []               # the pipe rows of the table being collected
    lines = text.split("\n")
    for line in lines:
        if in_fence is not None:
            m = _FENCE.match(line)
            if m and m.group(2)[0] == in_fence[0] and len(m.group(2)) >= len(in_fence):
                in_fence = None
                out.append(_span(_DIM, "└─", base))
                continue
            out.append(_span(_DIM, "│ ", base) + _span(_CODE, line, base))
            continue
        if _TABLE_ROW.match(line):
            table.append(line)
            continue
        if table:
            out.extend(_table(table, base))
            table = []
        m = _FENCE.match(line)
        if m:
            in_fence = m.group(2)
            label = "┌─"
            if m.group(3):
                label += " " + m.group(3)
            out.append(_span(_DIM, label, base))
            continue
        m = _HEADING.match(line)
        if m:
            style = _BOLD
            if len(m.group(1)) == 1:
                style = _BOLD + _UNDERLINE
            out.append(_span(style, m.group(2), base))
            continue
        if _RULE.match(line):
            out.append(_span(_DIM, "─" * max(1, min(width, 40)), base))
            continue
        m = _QUOTE.match(line)
        if m:
            out.append(m.group(1) + _span(_DIM, "▎ ", base) + _inline(m.group(2), base))
            continue
        m = _BULLET.match(line)
        if m:
            indent, body = m.group(1), m.group(2)
            task = _TASK.match(body)
            if task:
                box = "☑" if task.group(1) in "xX" else "☐"
                out.append(indent + box + " " + _inline(task.group(2), base))
            else:
                out.append(indent + "• " + _inline(body, base))
            continue
        m = _NUMBERED.match(line)
        if m:
            out.append(m.group(1) + m.group(2) + " " + _inline(m.group(3), base))
            continue
        out.append(_inline(line, base))
    if table:
        out.extend(_table(table, base))
    return "\n".join(out)


@cai.renderer("response")
def markdown(ctx: cai.RenderContext):
    return render_markdown(ctx.text, width=ctx.width, base=ctx.style)
