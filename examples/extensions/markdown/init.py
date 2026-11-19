"""init.py - example extension: render the assistant's markdown in the TUI.

a cai.renderer for the `response` kind: the transcript hands it the whole
assistant message so far (ctx.text) on every streamed chunk and once more when
the message ends (ctx.done), and paints what it returns in place of the raw
markdown.

two optional libraries do the real work, each used when it is importable in
cai's own environment (`pip install markdown-it-py pygments`, or the
`cai[markdown]` extra):

- markdown-it-py parses the message - a CommonMark parser, so emphasis,
  nesting, lists and tables follow the standard instead of a line regex. it
  takes an unfinished message as it is: an open code fence is still a code
  block, an unmatched `**` stays literal.
- Pygments colors fenced code by the language on the fence.

without markdown-it-py the block falls back to a line-oriented, stdlib-only
pass over the same constructs (render_plain); without Pygments code keeps one
color. nothing is required: the bundle loads and renders either way.

what it renders: headings (bold, # stripped), bullet and task lists (•, ☐, ☑),
numbered lists, block quotes (a bar), horizontal rules, fenced code (a framed
block, the language labelled), pipe tables (columns aligned), and inline
**bold**, *italic*, ~~strike~~, `code` and [links](url).

every styled span re-opens ctx.style (the block's base color) after its reset,
so the text around it keeps the transcript's assistant color. width is a
hint: tables and rules are laid out against it, everything else wraps on the
screen."""
import functools
import re
import unicodedata

import cai

try:
    from markdown_it import MarkdownIt
    from markdown_it.tree import SyntaxTreeNode
except ImportError:                      # optional: the plain pass renders instead
    MarkdownIt = None
    SyntaxTreeNode = None

try:
    from pygments.lexers import get_lexer_by_name
    from pygments.token import Token
except ImportError:                      # optional: code keeps a single color
    get_lexer_by_name = None
    Token = None


_ESC = "\033["
_RESET = _ESC + "m"
_BOLD = _ESC + "1m"
_DIM = _ESC + "2m"
_ITALIC = _ESC + "3m"
_UNDERLINE = _ESC + "4m"
_STRIKE = _ESC + "9m"
_CODE = _ESC + "38;5;215m"       # inline code, and fenced code with no lexer
_LINK = _ESC + "38;5;75m"        # link text

_ANSI = re.compile(r"\033\[[0-9;]*m")


def _span(style, text, base):
    """one styled span, closing back to the block's base style."""
    return style + text + _RESET + base


def _width(text):
    """the terminal columns a rendered string takes: escapes are free, wide
    (CJK, emoji) characters take two, combining marks none."""
    columns = 0
    for ch in _ANSI.sub("", text):
        if unicodedata.combining(ch):
            continue
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            columns += 2
            continue
        columns += 1
    return columns


# --- fenced code ------------------------------------------------------------

# pygments token type -> color. a token takes the entry of its nearest listed
# ancestor (Token.Name.Function.Magic -> Name.Function), else _CODE_TEXT.
_CODE_TEXT = _ESC + "38;5;252m"
_TOKEN_STYLES = {}
if Token is not None:
    _TOKEN_STYLES[Token.Comment] = _ESC + "38;5;245m"
    _TOKEN_STYLES[Token.Keyword] = _ESC + "38;5;176m"
    _TOKEN_STYLES[Token.Keyword.Constant] = _ESC + "38;5;215m"
    _TOKEN_STYLES[Token.Operator.Word] = _ESC + "38;5;176m"
    _TOKEN_STYLES[Token.String] = _ESC + "38;5;114m"
    _TOKEN_STYLES[Token.Number] = _ESC + "38;5;215m"
    _TOKEN_STYLES[Token.Name.Function] = _ESC + "38;5;75m"
    _TOKEN_STYLES[Token.Name.Class] = _ESC + "38;5;75m"
    _TOKEN_STYLES[Token.Name.Builtin] = _ESC + "38;5;80m"
    _TOKEN_STYLES[Token.Name.Decorator] = _ESC + "38;5;221m"
    _TOKEN_STYLES[Token.Name.Tag] = _ESC + "38;5;176m"
    _TOKEN_STYLES[Token.Name.Attribute] = _ESC + "38;5;221m"
    _TOKEN_STYLES[Token.Generic.Inserted] = _ESC + "38;5;114m"
    _TOKEN_STYLES[Token.Generic.Deleted] = _ESC + "38;5;203m"
    _TOKEN_STYLES[Token.Generic.Heading] = _BOLD
    _TOKEN_STYLES[Token.Generic.Subheading] = _ESC + "38;5;75m"


def _token_style(ttype):
    while ttype is not None:
        style = _TOKEN_STYLES.get(ttype)
        if style is not None:
            return style
        ttype = ttype.parent
    return _CODE_TEXT


@functools.lru_cache(maxsize=64)
def _lexer(lang):
    """the pygments lexer for a fence's language label, or None when pygments
    is missing or does not know the name. newlines are left exactly as
    written - the block is rendered line for line."""
    if get_lexer_by_name is None or not lang:
        return None
    try:
        return get_lexer_by_name(lang, stripnl=False, ensurenl=False)
    except Exception:                    # ClassNotFound, or a broken plugin lexer
        return None


@functools.lru_cache(maxsize=256)
def _code_lines(lang, code):
    """the lines of a code block, colored: by pygments when it knows `lang`,
    else in the one code color. each line is self-contained (every span closed
    with a reset), so the caller can frame them. cached - a streaming message
    re-renders per chunk, and its finished blocks never change."""
    if code.endswith("\n"):
        code = code[:-1]
    if not code:
        return ()
    lexer = _lexer(lang.lower())
    if lexer is None:
        plain = []
        for line in code.split("\n"):
            plain.append(_CODE + line + _RESET)
        return tuple(plain)
    lines = [[]]             # per line: [style, text] runs, same-style neighbours merged
    try:
        for ttype, value in lexer.get_tokens(code):
            style = _token_style(ttype)
            for i, part in enumerate(value.split("\n")):
                if i:
                    lines.append([])
                if not part:
                    continue
                runs = lines[-1]
                if runs and runs[-1][0] == style:
                    runs[-1][1] += part
                    continue
                runs.append([style, part])
    except Exception:                    # a lexer bug must not lose the block
        return _code_lines("", code)
    out = []
    for runs in lines:
        parts = []
        for style, text in runs:
            parts.append(style + text + _RESET)
        out.append("".join(parts))
    return tuple(out)


def _code_block(info, code, base, closed=True):
    """a framed code block: the language on the top edge, each line behind a
    bar, the bottom edge once the fence is closed (an open one is still
    streaming)."""
    words = (info or "").split()
    lang = ""
    if words:
        lang = words[0]
    label = "┌─"
    if lang:
        label += " " + lang
    out = [_span(_DIM, label, base)]
    for line in _code_lines(lang, code):
        out.append(_span(_DIM, "│ ", base) + line + base)
    if closed:
        out.append(_span(_DIM, "└─", base))
    return out


# --- the parsed pass (markdown-it-py) -----------------------------------------

_PARSER = None
if MarkdownIt is not None:
    # CommonMark plus pipe tables and ~~strike~~. setext headings are off: a
    # model writes '#' headings, and 'text\n---' is meant as a rule under a
    # line, not as that line turned into a heading.
    _PARSER = MarkdownIt("commonmark").enable("table").enable("strikethrough").disable("lheading")

_TASK_PREFIX = re.compile(r"^\[([ xX])\]\s+")


def _styled(text, styles, base):
    if not styles or not text:
        return text
    return styles + text + _RESET + base


def _inline(node, base, styles=""):
    """the inline children of `node`, rendered. `styles` is what the enclosing
    spans (a heading's bold, an outer **strong**) already turned on: each text
    run is emitted under the full stack and closed back to base, so nested
    emphasis composes and nothing bleeds past its span."""
    out = []
    for child in node.children:
        kind = child.type
        if kind == "text":
            out.append(_styled(child.content, styles, base))
        elif kind == "code_inline":
            out.append(_span(styles + _CODE, child.content, base))
        elif kind == "strong":
            out.append(_inline(child, base, styles + _BOLD))
        elif kind == "em":
            out.append(_inline(child, base, styles + _ITALIC))
        elif kind == "s":
            out.append(_inline(child, base, styles + _STRIKE))
        elif kind == "link":
            text = _inline(child, base, styles + _LINK + _UNDERLINE)
            href = str(child.attrs.get("href") or "")
            out.append(text)
            shown = _ANSI.sub("", text)
            if href and href != shown and href != "mailto:" + shown:
                out.append(_span(_DIM, f" ({href})", base))
        elif kind == "image":
            alt = child.content or "image"
            src = str(child.attrs.get("src") or "")
            out.append(_styled(alt, styles + _LINK, base))
            if src:
                out.append(_span(_DIM, f" ({src})", base))
        elif kind in ("softbreak", "hardbreak"):
            out.append("\n")
        elif child.children:
            out.append(_inline(child, base, styles))
        else:
            out.append(_styled(child.content, styles, base))
    return "".join(out)


def _inline_of(node, base, styles=""):
    """the rendered text of a block that wraps one inline run (a paragraph, a
    heading, a table cell)."""
    parts = []
    for child in node.children:
        if child.type == "inline":
            parts.append(_inline(child, base, styles))
    return "".join(parts)


def _blank_before(node, source):
    """whether the source put a blank line ahead of this block - the spacing
    is the author's, kept as written rather than normalized."""
    if node.map is None:
        return False
    start = node.map[0]
    if start <= 0 or start > len(source):
        return False
    return source[start - 1].strip() == ""


def _blocks(nodes, source, base, width):
    out = []
    for i, node in enumerate(nodes):
        if i and _blank_before(node, source):
            out.append("")
        out.extend(_block(node, source, base, width))
    return out


def _task_box(item):
    """'☐' / '☑' when the list item opens with a '[ ]' / '[x]' task marker
    (the marker is then cut from its text), else None."""
    if not item.children: return None
    first = item.children[0]
    if first.type != "paragraph" or not first.children: return None
    inline = first.children[0]
    if inline.type != "inline" or not inline.children: return None
    text = inline.children[0]
    if text.type != "text": return None
    m = _TASK_PREFIX.match(text.content)
    if m is None: return None
    text.token.content = text.content[m.end():]
    if m.group(1) in "xX":
        return "☑"
    return "☐"


def _list(node, source, base, width):
    ordered = node.type == "ordered_list"
    number = 1
    if ordered:
        try:
            number = int(node.attrs.get("start", 1))
        except (TypeError, ValueError):
            number = 1
    markers = []
    for item in node.children:
        if ordered:
            markers.append(f"{number}{node.markup or '.'}")
            number += 1
            continue
        markers.append(_task_box(item) or "•")
    marker_width = 0
    for marker in markers:
        marker_width = max(marker_width, len(marker))
    out = []
    for i, item in enumerate(node.children):
        if i and _blank_before(item, source):
            out.append("")
        lead = markers[i].rjust(marker_width) + " "
        hang = " " * len(lead)
        body = _blocks(item.children, source, base, max(1, width - len(lead)))
        if not body:
            out.append(lead.rstrip())
            continue
        for j, line in enumerate(body):
            if j == 0:
                out.append(lead + line)
            elif line:
                out.append(hang + line)
            else:
                out.append("")
    return out


def _pad(text, columns, align):
    gap = columns - _width(text)
    if gap <= 0:
        return text
    if align == "right":
        return " " * gap + text
    if align == "center":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


def _table(node, base):
    """a pipe table laid out in aligned columns: the header bold, a rule under
    it, cells joined by a dim bar. widths are measured on the rendered cells,
    so `code` or **bold** inside one does not push its column out of line."""
    rows = []                # (is_header, [rendered cell])
    aligns = []
    for section in node.children:
        header = section.type == "thead"
        for row in section.children:
            cells = []
            for j, cell in enumerate(row.children):
                styles = ""
                if header:
                    styles = _BOLD
                cells.append(_inline_of(cell, base, styles).replace("\n", " "))
                if header:
                    style = str(cell.attrs.get("style") or "")
                    align = "left"
                    if "right" in style: align = "right"
                    if "center" in style: align = "center"
                    while len(aligns) <= j:
                        aligns.append("left")
                    aligns[j] = align
            rows.append((header, cells))
    columns = 0
    for _header, cells in rows:
        columns = max(columns, len(cells))
    while len(aligns) < columns:
        aligns.append("left")
    widths = [0] * columns
    for _header, cells in rows:
        for j, cell in enumerate(cells):
            widths[j] = max(widths[j], _width(cell))
    bar = _span(_DIM, " │ ", base)
    out = []
    ruled = False
    for header, cells in rows:
        if not header and not ruled:
            ruled = True
            rule = []
            for w in widths:
                rule.append("─" * w)
            out.append(_span(_DIM, "─┼─".join(rule), base))
        padded = []
        for j in range(columns):
            cell = ""
            if j < len(cells):
                cell = cells[j]
            padded.append(_pad(cell, widths[j], aligns[j]))
        out.append(bar.join(padded).rstrip())
    return out


def _fence_closed(node, source):
    """whether a fenced block has its closing fence yet - while the message
    streams the last block is usually still open."""
    if node.map is None:
        return True
    start, end = node.map
    if end - start < 2 or end > len(source):
        return False
    last = source[end - 1].strip()
    marker = node.markup or "```"
    return last.startswith(marker) and last.strip(marker[0]) == ""


def _block(node, source, base, width):
    kind = node.type
    if kind == "paragraph":
        return _inline_of(node, base).split("\n")
    if kind == "heading":
        styles = _BOLD
        if node.tag == "h1":
            styles = _BOLD + _UNDERLINE
        return _inline_of(node, base, styles).split("\n")
    if kind in ("bullet_list", "ordered_list"):
        return _list(node, source, base, width)
    if kind == "blockquote":
        out = []
        for line in _blocks(node.children, source, base, max(1, width - 2)):
            out.append(_span(_DIM, "▎ ", base) + line)
        return out
    if kind == "fence":
        return _code_block(node.info, node.content, base, _fence_closed(node, source))
    if kind == "code_block":
        return _code_block("", node.content, base)
    if kind == "hr":
        return [_span(_DIM, "─" * max(1, min(width, 40)), base)]
    if kind == "table":
        return _table(node, base)
    if node.children:
        return _blocks(node.children, source, base, width)
    return (node.content or "").rstrip("\n").split("\n")


def render_parsed(text, width=80, base=""):
    """the standard pass: parse with markdown-it-py, render its tree."""
    source = text.split("\n")
    tree = SyntaxTreeNode(_PARSER.parse(text))
    out = "\n".join(_blocks(tree.children, source, base, width))
    if text.endswith("\n"):
        out += "\n"
    return out


# --- the plain pass (stdlib only) ---------------------------------------------

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


def _plain_inline(text, base):
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


def _plain_table(rows, base):
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
            text = _plain_inline(cell, base)
            if rule_at is not None and i < rule_at:
                text = _span(_BOLD, cell, base)
            padded.append(text + pad)
        out.append(bar.join(padded).rstrip())
    return out


def render_plain(text, width=80, base=""):
    """the fallback pass, used when markdown-it-py is not installed: one walk
    over the lines - fences, tables, headings, rules, quotes, lists - then the
    inline regexes on everything else. returns the rendered text."""
    out = []
    in_fence = None          # the fence marker while inside a code block
    fence_lang = ""
    fence_code = []          # the block's lines so far
    table = []               # the pipe rows of the table being collected
    lines = text.split("\n")
    for line in lines:
        if in_fence is not None:
            m = _FENCE.match(line)
            if m and m.group(2)[0] == in_fence[0] and len(m.group(2)) >= len(in_fence):
                out.extend(_code_block(fence_lang, "\n".join(fence_code) + "\n", base))
                in_fence = None
                continue
            fence_code.append(line)
            continue
        if _TABLE_ROW.match(line):
            table.append(line)
            continue
        if table:
            out.extend(_plain_table(table, base))
            table = []
        m = _FENCE.match(line)
        if m:
            in_fence = m.group(2)
            fence_lang = m.group(3)
            fence_code = []
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
            out.append(m.group(1) + _span(_DIM, "▎ ", base) + _plain_inline(m.group(2), base))
            continue
        m = _BULLET.match(line)
        if m:
            indent, body = m.group(1), m.group(2)
            task = _TASK.match(body)
            if task:
                box = "☑" if task.group(1) in "xX" else "☐"
                out.append(indent + box + " " + _plain_inline(task.group(2), base))
            else:
                out.append(indent + "• " + _plain_inline(body, base))
            continue
        m = _NUMBERED.match(line)
        if m:
            out.append(m.group(1) + m.group(2) + " " + _plain_inline(m.group(3), base))
            continue
        out.append(_plain_inline(line, base))
    if in_fence is not None:
        out.extend(_code_block(fence_lang, "\n".join(fence_code), base, closed=False))
    if table:
        out.extend(_plain_table(table, base))
    return "\n".join(out)


def render_markdown(text, width=80, base=""):
    """render one message: the parsed pass when markdown-it-py is installed,
    the plain pass otherwise."""
    if _PARSER is not None:
        return render_parsed(text, width, base)
    return render_plain(text, width, base)


@cai.renderer("response")
def markdown(ctx: cai.RenderContext):
    return render_markdown(ctx.text, width=ctx.width, base=ctx.style)
