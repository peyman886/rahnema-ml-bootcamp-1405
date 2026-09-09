"""Reference figures drawn with matplotlib.

Why matplotlib instead of a PNG someone made in Figma: the figures stay in
version control as code, they scale to any projector resolution, and the
numbers on them are the same numbers the notebook cells print.

Every function returns the Figure, and optionally writes it to disk.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# A calm palette that survives a badly calibrated projector: dark ink on
# white, with light fills and saturated borders for the highlights.
INK = "#1f2933"
MUTED = "#616e7c"
BORDER = "#9aa5b1"

ACCENTS = {
    "blue": ("#dbeafe", "#2563eb"),
    "orange": ("#ffedd5", "#ea580c"),
    "green": ("#d1fae5", "#059669"),
    "purple": ("#ede9fe", "#7c3aed"),
    "red": ("#fee2e2", "#dc2626"),
    "grey": ("#f1f5f9", "#94a3b8"),
}

MONO = {"family": "monospace"}


# --------------------------------------------------------------------------
# low-level drawing helpers
# --------------------------------------------------------------------------

def _panel(ax, title, xlim, ylim):
    """Turn a normal Axes into a blank drawing surface with a code-ish title."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=10, loc="left")


def _grid(ax, values, x0=0.0, y0=0.0, cell=1.0, fills=None, edges=None,
          fontsize=10, lw=1.2, fmt=str):
    """Draw a 2-D array of labels as a table of boxes, row 0 on top.

    ``fills``/``edges`` accept either a single colour or an array shaped like
    ``values``, which is how the highlights below are done.
    """
    values = np.atleast_2d(np.asarray(values, dtype=object))
    n_rows, n_cols = values.shape

    def pick(spec, r, c, default):
        if spec is None:
            return default
        arr = np.atleast_2d(np.asarray(spec, dtype=object))
        if arr.shape == values.shape:
            return arr[r, c]
        return spec

    for r in range(n_rows):
        for c in range(n_cols):
            x = x0 + c * cell
            y = y0 - (r + 1) * cell
            ax.add_patch(Rectangle(
                (x, y), cell, cell,
                facecolor=pick(fills, r, c, "white"),
                edgecolor=pick(edges, r, c, BORDER),
                linewidth=lw, zorder=2,
            ))
            label = values[r, c]
            if label is not None and label != "":
                ax.text(x + cell / 2, y + cell / 2, fmt(label),
                        ha="center", va="center", fontsize=fontsize,
                        color=INK, zorder=3, **MONO)
    return x0 + n_cols * cell, y0 - n_rows * cell  # bottom-right corner


def _arrow(ax, start, end, color=MUTED, lw=1.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=14,
        color=color, linewidth=lw, linestyle=ls, zorder=4,
        shrinkA=2, shrinkB=2,
    ))


def _code(ax, x, y, text, color=INK, fontsize=10.5, ha="left", va="center",
          weight="normal"):
    return ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
                   fontweight=weight, zorder=5, **MONO)


def _note(ax, x, y, text, color=MUTED, fontsize=10, ha="left", va="center",
          style="italic"):
    return ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
                   fontstyle=style, zorder=5)


def _save(fig, save_to):
    if save_to:
        fig.savefig(save_to, dpi=170, bbox_inches="tight", facecolor="white")
    return fig


# --------------------------------------------------------------------------
# NumPy
# --------------------------------------------------------------------------

# Every panel is drawn inside the same 13 x 11 box so that all six end up at
# the same visual scale.
_W, _H = 13.0, 11.0


def _np_axes_panel(ax):
    _panel(ax, "1 · shape & axes", (0, _W), (0, _H))
    A = np.arange(12).reshape(3, 4)
    _grid(ax, A, x0=2.6, y0=8.2, cell=1.6, fontsize=12)

    _arrow(ax, (1.9, 8.0), (1.9, 3.6))
    ax.text(1.3, 5.8, "axis=0", rotation=90, ha="center", va="center",
            fontsize=11, color=MUTED, **MONO)
    _arrow(ax, (2.8, 8.9), (8.8, 8.9))
    _code(ax, 5.8, 9.6, "axis=1", color=MUTED, fontsize=11, ha="center")

    _code(ax, 0.6, 2.5, "A = np.arange(12).reshape(3, 4)")
    _code(ax, 0.6, 1.5, "A.shape -> (3, 4)    A.ndim -> 2    A.size -> 12",
          color=MUTED, fontsize=9.5)
    _note(ax, 0.6, 0.6, "Print .shape after every step. It is the cheapest debugger you have.")


def _np_indexing_panel(ax):
    _panel(ax, "2 · indexing & slicing", (0, _W), (0, _H))
    A = np.arange(12).reshape(3, 4)
    cell = 0.82

    cases = [
        ("A[0]", np.s_[0], "(4,)", "blue"),
        ("A[:, 0]", np.s_[:, 0], "(3,)", "orange"),
        ("A[1:, ::2]", np.s_[1:, ::2], "(2, 2)", "green"),
    ]
    for i, (label, sl, out_shape, colour) in enumerate(cases):
        light, dark = ACCENTS[colour]
        x0 = 0.7 + i * 4.3
        # Mark which cells the slice touches by writing into a mask.
        hit = np.zeros_like(A, dtype=bool)
        hit[sl] = True
        fills = np.where(hit, light, "white")
        edges = np.where(hit, dark, BORDER)
        _grid(ax, A, x0=x0, y0=8.4, cell=cell, fills=fills, edges=edges, fontsize=9)
        _code(ax, x0, 9.1, label, color=dark, fontsize=11, weight="bold")
        _code(ax, x0, 5.5, "-> " + out_shape, color=MUTED, fontsize=9.5)

    lines = [
        ("A[-1]", "last row"),
        ("A[[0, 2]]", "pick rows 0 and 2 (fancy indexing)"),
        ("A[A > 6]", "boolean mask -> flat 1-D result"),
        ("A[0, 1] = 99", "assignment works the same way"),
    ]
    for i, (code, what) in enumerate(lines):
        y = 4.3 - i * 0.85
        _code(ax, 0.7, y, code, fontsize=9.5)
        _note(ax, 5.0, y, what, fontsize=9.5, style="normal")

    _code(ax, 0.7, 0.5, "slices share memory (view) · fancy/boolean indexing copies",
          color=ACCENTS["red"][1], fontsize=9.5)


def _np_reduce_panel(ax):
    _panel(ax, "3 \u00b7 reductions: pick the axis to kill", (0, _W), (0, _H))
    A = np.arange(12).reshape(3, 4)
    cell = 1.3
    _grid(ax, A, x0=2.2, y0=9.8, cell=cell, fontsize=11)

    or_l, or_d = ACCENTS["orange"]
    _grid(ax, A.sum(axis=1)[:, None], x0=8.2, y0=9.8, cell=cell,
          fills=or_l, edges=or_d, fontsize=11)
    _arrow(ax, (7.6, 7.85), (8.15, 7.85), color=or_d)
    _code(ax, 8.85, 10.2, "axis=1", color=or_d, fontsize=10, ha="center")
    _code(ax, 8.85, 5.4, "-> (3,)", color=or_d, fontsize=10, ha="center")

    blue_l, blue_d = ACCENTS["blue"]
    _grid(ax, A.sum(axis=0)[None, :], x0=2.2, y0=4.9, cell=cell,
          fills=blue_l, edges=blue_d, fontsize=11)
    _arrow(ax, (4.8, 5.8), (4.8, 5.0), color=blue_d)
    _code(ax, 2.2, 3.1, "A.sum(axis=0) -> (4,)", color=blue_d, fontsize=10)

    _note(ax, 0.4, 2.0, "\"axis=k\" means: collapse axis k, it leaves the shape.")
    _code(ax, 0.4, 1.2, "A.sum()  A.mean(axis=0)  A.std()  A.argmax()", fontsize=9.5)
    _code(ax, 0.4, 0.4, "keepdims=True keeps it as size 1, for broadcasting",
          color=MUTED, fontsize=9)


def _np_mask_panel(ax):
    _panel(ax, "4 \u00b7 boolean masks = filtering", (0, _W), (0, _H))
    a = np.array([3, -1, 7, 0, -5, 2])
    cell, x0 = 1.2, 3.6
    gr_l, gr_d = ACCENTS["green"]
    grey_l, grey_d = ACCENTS["grey"]

    _grid(ax, a[None, :], x0=x0, y0=10.4, cell=cell, fontsize=11)
    _code(ax, x0 - 0.25, 9.8, "a", ha="right", fontsize=11, weight="bold")

    mask = a > 0
    _grid(ax, np.where(mask, "T", "F")[None, :], x0=x0, y0=8.4, cell=cell,
          fills=np.where(mask, gr_l, grey_l)[None, :],
          edges=np.where(mask, gr_d, grey_d)[None, :], fontsize=10)
    _code(ax, x0 - 0.25, 7.8, "a > 0", ha="right", fontsize=11, weight="bold")

    _grid(ax, a[mask][None, :], x0=x0, y0=6.4, cell=cell,
          fills=gr_l, edges=gr_d, fontsize=11)
    _code(ax, x0 - 0.25, 5.8, "a[a > 0]", ha="right", fontsize=11, weight="bold")

    for y_from, y_to in ((9.2, 8.5), (7.2, 6.5)):
        _arrow(ax, (x0 + 3 * cell, y_from), (x0 + 3 * cell, y_to))

    rows = [
        ("np.where(a > 0, a, 0)", "= ReLU, back in week 4"),
        ("a.clip(0)", "the same, shorter"),
        ("mask.sum() / mask.mean()", "count / fraction"),
        ("(a > 0) & (a < 5)", "use & | ~ , not and/or"),
    ]
    for i, (code, what) in enumerate(rows):
        y = 4.2 - i * 0.95
        _code(ax, 0.4, y, code, fontsize=9)
        _note(ax, 6.4, y, what, fontsize=9, style="normal")

    _note(ax, 0.4, 0.3, "A mask is just an array of booleans. Reuse it, count it, invert it.")


def _np_broadcast_panel(ax):
    _panel(ax, "5 \u00b7 broadcasting", (0, _W), (0, _H))
    rules = [
        "1. line the shapes up from the RIGHT",
        "2. each pair must be equal, or one must be 1",
        "3. size-1 axes are stretched for free (no copy)",
    ]
    for i, rule in enumerate(rules):
        _note(ax, 0.3, 10.3 - i * 0.8, rule, fontsize=9.5, style="normal")

    cases = [
        ("X - mu", ["  (5, 3)", "  (   3,)", "  --------", "  (5, 3)"], "green", "works"),
        ("X - col", ["  (5, 3)", "  (5,   )", "  --------", "  ERROR"], "red", "shapes clash"),
        ("X - col[:, None]", ["  (5, 3)", "  (5, 1)", "  --------", "  (5, 3)"], "green", "the fix"),
    ]
    for i, (title, block, colour, verdict) in enumerate(cases):
        light, dark = ACCENTS[colour]
        x = 0.3 + i * 4.25
        ax.add_patch(Rectangle((x, 2.6), 4.0, 5.4, facecolor=light,
                               edgecolor=dark, linewidth=1.2, zorder=1))
        _code(ax, x + 0.2, 7.5, title, color=dark, fontsize=9.5, weight="bold")
        for j, line in enumerate(block):
            _code(ax, x + 0.2, 6.6 - j * 0.85, line, fontsize=10)
        _note(ax, x + 0.2, 3.0, verdict, color=dark, fontsize=9)

    _code(ax, 0.3, 1.7, "mu = X.mean(axis=0) -> (3,)    col = X.mean(axis=1) -> (5,)",
          color=MUTED, fontsize=8.5)
    _note(ax, 0.3, 0.8, "[:, None] adds a size-1 axis. It is the everyday fix, not a trick.",
          fontsize=9.5)


def _np_reference_panel(ax):
    _panel(ax, "6 \u00b7 the rest, in one place", (0, _W), (0, _H))
    rows = [
        ("create", "np.zeros((3,4))  np.ones  np.full  np.eye  np.arange"),
        ("random", "rng = np.random.default_rng(0); rng.normal(size=(3,4))"),
        ("reshape", "A.reshape(2, 6)  A.reshape(-1)  A.T  A.ravel()"),
        ("newaxis", "A[:, None]  turns (n,) into (n, 1)"),
        ("join", "np.concatenate([A, B], axis=0)  np.stack  vstack"),
        ("linalg", "A @ B   np.linalg.solve  .svd  .norm  .eig  .cond"),
        ("sort", "np.sort(a)  np.argsort(a)  a.argmax()  np.unique(a)"),
        ("floats", "np.isclose(a, b)  -- never use == on floats"),
    ]
    for i, (label, code) in enumerate(rows):
        y = 9.8 - i * 1.22
        _code(ax, 0.3, y, label, color=ACCENTS["purple"][1], fontsize=9.5, weight="bold")
        _code(ax, 3.4, y, code, fontsize=9)

    _note(ax, 0.3, 0.2, "If you are writing a for-loop over an array, there is a one-liner.")


def numpy_cheatsheet(save_to=None):
    """One-page NumPy reference: shapes, indexing, reductions, broadcasting."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.2), facecolor="white")
    panels = [_np_axes_panel, _np_indexing_panel, _np_reduce_panel,
              _np_mask_panel, _np_broadcast_panel, _np_reference_panel]
    for ax, draw in zip(axes.ravel(), panels):
        draw(ax)
    fig.suptitle("NumPy in one page", fontsize=19, fontweight="bold", color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, save_to)


# --------------------------------------------------------------------------
# pandas
# --------------------------------------------------------------------------

def _table(ax, columns, index, body, x0, y0, cw=2.4, ch=0.9,
           index_w=1.2, fills=None, edges=None, fontsize=9.5,
           header_fill="#eef2f7", show_index=True):
    """Draw a DataFrame-shaped table: header row, index column, body cells.

    Returns (x_right, y_bottom) so callers can hang arrows off it.
    """
    body = np.atleast_2d(np.asarray(body, dtype=object))
    n_rows, n_cols = body.shape
    xb = x0 + (index_w if show_index else 0.0)
    head_h = ch if len(columns) else 0.0

    for c, name in enumerate(columns):
        ax.add_patch(Rectangle((xb + c * cw, y0 - ch), cw, ch,
                               facecolor=header_fill, edgecolor=BORDER,
                               linewidth=1.2, zorder=2))
        ax.text(xb + c * cw + cw / 2, y0 - ch / 2, str(name), ha="center",
                va="center", fontsize=fontsize, fontweight="bold",
                color=INK, zorder=3, **MONO)

    def pick(spec, r, c, default):
        if spec is None:
            return default
        arr = np.atleast_2d(np.asarray(spec, dtype=object))
        if arr.shape == body.shape:
            return arr[r, c]
        return spec

    for r in range(n_rows):
        y = y0 - head_h - (r + 1) * ch
        if show_index:
            ax.add_patch(Rectangle((x0, y), index_w, ch, facecolor=header_fill,
                                   edgecolor=BORDER, linewidth=1.2, zorder=2))
            ax.text(x0 + index_w / 2, y + ch / 2, str(index[r]), ha="center",
                    va="center", fontsize=fontsize, color=MUTED, zorder=3, **MONO)
        for c in range(n_cols):
            ax.add_patch(Rectangle((xb + c * cw, y), cw, ch,
                                   facecolor=pick(fills, r, c, "white"),
                                   edgecolor=pick(edges, r, c, BORDER),
                                   linewidth=1.2, zorder=2))
            ax.text(xb + c * cw + cw / 2, y + ch / 2, str(body[r, c]),
                    ha="center", va="center", fontsize=fontsize, color=INK,
                    zorder=3, **MONO)

    return xb + n_cols * cw, y0 - head_h - n_rows * ch


# The toy frame every pandas panel refers to. Small enough to draw, but it
# already contains a NaN and a repeated key, which is all the panels need.
_PD_COLS = ["city", "qty", "price"]
_PD_BODY = [["Tehran", 2, 35.0],
            ["Karaj", 1, 12.5],
            ["Tehran", 4, 80.0],
            ["Qom", "NaN", 22.0]]


def _pd_anatomy_panel(ax):
    _panel(ax, "1 \u00b7 what a DataFrame is made of", (0, _W), (0, _H))
    cw, ch, x0 = 2.4, 1.0, 1.8
    x_right, y_bottom = _table(ax, _PD_COLS, [0, 1, 2, 3], _PD_BODY,
                               x0=x0, y0=9.9, cw=cw, ch=ch, index_w=1.1,
                               fontsize=10)

    pl, pd_ = ACCENTS["purple"]
    _table(ax, [], [], [["object", "float64", "float64"]], x0=x0,
           y0=y_bottom, cw=cw, ch=ch, index_w=1.1, fills=pl, edges=pd_,
           show_index=False, fontsize=9)
    _code(ax, x0 - 0.25, y_bottom - ch / 2, ".dtypes", color=pd_, fontsize=10,
          weight="bold", ha="right")

    _arrow(ax, (1.15, 7.4), (1.75, 7.4), color=ACCENTS["orange"][1])
    _code(ax, 0.15, 8.1, "index", color=ACCENTS["orange"][1], fontsize=10, weight="bold")
    _arrow(ax, (x_right + 0.85, 9.4), (x_right + 0.15, 9.4), color=ACCENTS["blue"][1])
    _code(ax, x_right + 0.95, 9.4, "columns", color=ACCENTS["blue"][1], fontsize=10,
          weight="bold")
    _arrow(ax, (x_right + 0.85, 6.9), (x_right + 0.15, 6.9), color=ACCENTS["green"][1])
    _code(ax, x_right + 0.95, 6.9, "values", color=ACCENTS["green"][1], fontsize=10,
          weight="bold")

    _note(ax, 0.3, 2.6, "A DataFrame is a dict of Series that share one index.")
    _code(ax, 0.3, 1.7, 'df["price"] -> Series', fontsize=9.5)
    _code(ax, 0.3, 0.9, 'df[["city", "price"]] -> DataFrame', fontsize=9.5)
    _code(ax, 0.3, 0.1, "one column = one dtype. Mixed content collapses to object.",
          color=MUTED, fontsize=9)


def _pd_select_panel(ax):
    _panel(ax, "2 \u00b7 selecting rows & columns", (0, _W), (0, _H))
    cw, ch, x0 = 2.0, 0.72, 0.4
    bl, bd = ACCENTS["blue"]
    ol, od = ACCENTS["orange"]
    gl, gd = ACCENTS["green"]

    fills = np.full((4, 3), "white", dtype=object)
    edges = np.full((4, 3), BORDER, dtype=object)
    fills[:, 2], edges[:, 2] = bl, bd
    fills[1, 0], edges[1, 0] = ol, od
    _table(ax, _PD_COLS, [0, 1, 2, 3], _PD_BODY, x0=x0, y0=10.5, cw=cw, ch=ch,
           index_w=0.9, fills=fills, edges=edges, fontsize=9)
    _code(ax, 7.7, 9.4, 'df["price"]', color=bd, fontsize=10, weight="bold")
    _code(ax, 7.7, 8.6, "-> a Series", color=MUTED, fontsize=9)
    _code(ax, 7.7, 7.5, 'df.loc[1, "city"]', color=od, fontsize=10, weight="bold")
    _code(ax, 7.7, 6.7, "-> one value", color=MUTED, fontsize=9)

    fills2 = np.full((4, 3), "white", dtype=object)
    edges2 = np.full((4, 3), BORDER, dtype=object)
    for r in (0, 2):
        fills2[r, :], edges2[r, :] = gl, gd
    _table(ax, _PD_COLS, [0, 1, 2, 3], _PD_BODY, x0=x0, y0=5.8, cw=cw, ch=ch,
           index_w=0.9, fills=fills2, edges=edges2, fontsize=9)
    _code(ax, 7.7, 4.6, 'df[df["qty"] > 1]', color=gd, fontsize=10, weight="bold")
    _code(ax, 7.7, 3.8, "-> a DataFrame", color=MUTED, fontsize=9)

    _code(ax, 0.3, 1.5, ".loc = by label (end included)    .iloc = by position",
          fontsize=9)
    _code(ax, 0.3, 0.75, 'df.loc[df.qty > 1, "price"] = 0   assigns',
          color=ACCENTS["green"][1], fontsize=9)
    _code(ax, 0.3, 0.0, 'df[df.qty > 1]["price"] = 0       silently does nothing',
          color=ACCENTS["red"][1], fontsize=9)


def _pd_firstlook_panel(ax):
    _panel(ax, "3 · the first-look ritual", (0, _W), (0, _H))
    steps = [
        ("df.shape", "how many rows and columns"),
        ("df.dtypes", "a number stored as object means text got in"),
        ("df.head(), df.sample(5)", "read real rows, not just column names"),
        ("df.isna().mean()", "missing share per column"),
        ('df.describe(include="all")', "ranges, outliers, cardinality"),
        ("df.duplicated().sum()", "duplicate rows"),
        ("df[c].value_counts(dropna=False)", "for every category column"),
    ]
    for i, (code, why) in enumerate(steps):
        y = 9.9 - i * 1.28
        ax.add_patch(Rectangle((0.3, y - 0.42), 0.85, 0.85,
                               facecolor=ACCENTS["blue"][0],
                               edgecolor=ACCENTS["blue"][1], linewidth=1.1, zorder=2))
        ax.text(0.72, y, str(i + 1), ha="center", va="center", fontsize=10,
                color=ACCENTS["blue"][1], fontweight="bold", zorder=3, **MONO)
        _code(ax, 1.5, y + 0.22, code, fontsize=9.5)
        _note(ax, 1.5, y - 0.35, why, fontsize=9, style="normal")

    _code(ax, 0.3, 0.6, "dtypes before head(). A numeric column typed as text",
          color=ACCENTS["red"][1], fontsize=9.5)
    _code(ax, 0.3, 0.0, "makes every later number wrong, and it never warns you.",
          color=ACCENTS["red"][1], fontsize=9.5)


def _pd_groupby_panel(ax):
    _panel(ax, "4 \u00b7 groupby = split \u00b7 apply \u00b7 combine", (0, _W), (0, _H))
    ch = 0.78
    keys = ["Tehran", "Karaj", "Tehran", "Qom"]
    prices = [35.0, 12.5, 80.0, 22.0]
    colour_of = {"Tehran": "blue", "Karaj": "orange", "Qom": "green"}

    fills = np.array([[ACCENTS[colour_of[k]][0]] * 2 for k in keys], dtype=object)
    edges = np.array([[ACCENTS[colour_of[k]][1]] * 2 for k in keys], dtype=object)
    _table(ax, ["city", "price"], [0, 1, 2, 3],
           [[k, p] for k, p in zip(keys, prices)],
           x0=0.2, y0=9.4, cw=1.55, ch=ch, index_w=0.7, fills=fills, edges=edges,
           fontsize=8.5)
    _code(ax, 0.2, 9.8, "df", fontsize=10, weight="bold")

    y = 9.4
    for city in ("Tehran", "Karaj", "Qom"):
        rows = [[city, p] for k, p in zip(keys, prices) if k == city]
        light, dark = ACCENTS[colour_of[city]]
        _table(ax, [], [], rows, x0=5.0, y0=y, cw=1.55, ch=ch,
               fills=light, edges=dark, show_index=False, fontsize=8.5)
        y -= (len(rows) + 0.55) * ch
    _code(ax, 5.0, 9.8, "split by key", fontsize=10, weight="bold")
    _arrow(ax, (4.2, 7.4), (4.9, 7.4))

    res = [["Karaj", 12.5, 1], ["Qom", 22.0, 1], ["Tehran", 57.5, 2]]
    fills3 = np.array([[ACCENTS[colour_of[r[0]]][0]] * 3 for r in res], dtype=object)
    edges3 = np.array([[ACCENTS[colour_of[r[0]]][1]] * 3 for r in res], dtype=object)
    _table(ax, ["city", "median", "count"], [], res, x0=8.9, y0=9.4, cw=1.3,
           ch=ch, fills=fills3, edges=edges3, show_index=False, fontsize=8)
    _code(ax, 8.9, 9.8, "apply + combine", fontsize=10, weight="bold")
    _arrow(ax, (8.2, 8.2), (8.8, 8.2))

    _code(ax, 0.2, 2.6, 'df.groupby("city")["price"]', fontsize=9.5)
    _code(ax, 0.2, 1.8, '      .agg(["median", "count"])', fontsize=9.5)
    _code(ax, 0.2, 1.0, 'df.groupby(["city", "year"])["price"].median().unstack()',
          fontsize=8.5)
    _note(ax, 0.2, 0.2, "transform() returns it at row level -- great for group features.",
          fontsize=9)


def _pd_merge_panel(ax):
    _panel(ax, "5 \u00b7 merge = SQL join", (0, _W), (0, _H))
    ch = 0.8
    bl, bd = ACCENTS["blue"]
    ol, od = ACCENTS["orange"]

    _table(ax, ["city", "price"], [0, 1, 2],
           [["Tehran", 35.0], ["Karaj", 12.5], ["Qom", 22.0]],
           x0=0.2, y0=10.2, cw=1.7, ch=ch, index_w=0.8, fills=bl, edges=bd,
           fontsize=9)
    _code(ax, 0.2, 10.6, "orders", color=bd, fontsize=10, weight="bold")

    _table(ax, ["city", "region"], [0, 1],
           [["Tehran", "center"], ["Karaj", "center"]],
           x0=6.4, y0=10.2, cw=1.9, ch=ch, index_w=0.8, fills=ol, edges=od,
           fontsize=9)
    _code(ax, 6.4, 10.6, "city_info", color=od, fontsize=10, weight="bold")

    fills = np.array([[bl, bl, ol], [bl, bl, ol],
                      [bl, bl, ACCENTS["red"][0]]], dtype=object)
    _table(ax, ["city", "price", "region"], [0, 1, 2],
           [["Tehran", 35.0, "center"], ["Karaj", 12.5, "center"],
            ["Qom", 22.0, "NaN"]],
           x0=1.8, y0=6.0, cw=2.0, ch=ch, index_w=0.8, fills=fills, fontsize=9)
    _code(ax, 1.8, 6.4, 'how="left" keeps every order', fontsize=10, weight="bold")
    _arrow(ax, (7.6, 7.1), (7.6, 6.2))

    _code(ax, 0.2, 2.4, 'df.merge(city_info, on="city",', fontsize=9.5)
    _code(ax, 0.2, 1.7, '         how="left", validate="m:1")', fontsize=9.5)
    _code(ax, 0.2, 0.9, "left / inner / outer / right", color=MUTED, fontsize=9.5)
    _code(ax, 0.2, 0.1, "rows grew after a merge? your key was not unique.",
          color=ACCENTS["red"][1], fontsize=9.5)


def _pd_reference_panel(ax):
    _panel(ax, "6 \u00b7 the rest, in one place", (0, _W), (0, _H))
    rows = [
        ("read", 'pd.read_csv(p, parse_dates=["d"])   df.to_parquet(p)'),
        ("new col", 'df["ppq"] = df["price"] / df["qty"]   df.assign(...)'),
        ("missing", "df.isna().sum()   .dropna(subset=[...])   .fillna(0)"),
        ("category", 's.value_counts()   s.nunique()   s.astype("category")'),
        ("map", 's.map({"a": 1})   np.where(cond, x, y)   s.replace()'),
        ("sort", 'df.sort_values("price")   df.nlargest(5, "price")'),
        ("reshape", "df.pivot_table(index=, columns=, values=, aggfunc=)"),
        ("time", 'pd.to_datetime(s)   df.resample("D").sum()   s.dt.hour'),
        ("strings", "s.str.strip().str.lower()   s.str.contains(...)"),
    ]
    for i, (label, code) in enumerate(rows):
        y = 9.9 - i * 1.1
        _code(ax, 0.3, y, label, color=ACCENTS["purple"][1], fontsize=9, weight="bold")
        _code(ax, 3.1, y, code, fontsize=8.5)

    _note(ax, 0.3, 0.1, "Chain method calls. Never a for-loop over df.iterrows().",
          fontsize=9)


def pandas_cheatsheet(save_to=None):
    """One-page pandas reference: anatomy, selection, groupby, merge, reshape."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.2), facecolor="white")
    panels = [_pd_anatomy_panel, _pd_select_panel, _pd_firstlook_panel,
              _pd_groupby_panel, _pd_merge_panel, _pd_reference_panel]
    for ax, draw in zip(axes.ravel(), panels):
        draw(ax)
    fig.suptitle("pandas in one page", fontsize=19, fontweight="bold", color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, save_to)


# --------------------------------------------------------------------------
# Data science pipeline
# --------------------------------------------------------------------------

# (number, name, one-line job, where the bootcamp covers it, colour key)
_STAGES = [
    ("1", "Problem definition", "what decision changes?", "every week", "grey"),
    ("2", "Data collection", "where does it live?", "wk 9 data stack", "grey"),
    ("3", "Cleaning & preprocessing", "make it usable", "today + wk 3", "green"),
    ("4", "Exploratory analysis", "understand before modelling", "today", "green"),
    ("5", "Modeling", "fit something, honestly", "wk 3 - 5", "blue"),
    ("6", "Evaluation", "is it actually good?", "wk 3 + wk 7", "blue"),
    ("7", "Deployment", "ship it", "wk 9", "purple"),
    ("8", "Monitoring", "it will rot", "wk 9", "purple"),
    ("9", "Reporting", "make someone act", "wk 11", "orange"),
]


def pipeline_diagram(save_to=None):
    """The nine stages, drawn as the loop it really is."""
    fig, ax = plt.subplots(figsize=(16, 7.4), facecolor="white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0.0, 8.9)
    ax.axis("off")

    bw, bh = 2.75, 1.7
    top_y, bot_y = 5.3, 1.6
    xs_top = [0.35 + i * 3.12 for i in range(5)]
    xs_bot = [0.35 + i * 3.12 for i in (4, 3, 2, 1)]

    boxes = {}
    for i, (num, name, job, when, colour) in enumerate(_STAGES):
        x = xs_top[i] if i < 5 else xs_bot[i - 5]
        y = top_y if i < 5 else bot_y
        light, dark = ACCENTS[colour]
        ax.add_patch(Rectangle((x, y), bw, bh, facecolor=light, edgecolor=dark,
                               linewidth=1.8, zorder=3))
        ax.text(x + 0.22, y + bh - 0.34, num, fontsize=11, color=dark,
                fontweight="bold", va="center", zorder=4, **MONO)
        ax.text(x + bw / 2, y + bh - 0.62, name, fontsize=11.5, color=INK,
                fontweight="bold", ha="center", va="center", zorder=4)
        ax.text(x + bw / 2, y + 0.62, job, fontsize=9.5, color=MUTED,
                ha="center", va="center", fontstyle="italic", zorder=4)
        ax.text(x + bw / 2, y + 0.22, when, fontsize=8.5, color=dark,
                ha="center", va="center", zorder=4, **MONO)
        boxes[i] = (x, y)

    for i in range(4):  # left to right along the top
        _arrow(ax, (xs_top[i] + bw, top_y + bh / 2), (xs_top[i + 1], top_y + bh / 2),
               color=MUTED, lw=2.0)
    _arrow(ax, (xs_top[4] + bw / 2, top_y), (xs_top[4] + bw / 2, bot_y + bh),
           color=MUTED, lw=2.0)
    for i in range(3):  # right to left along the bottom
        _arrow(ax, (xs_bot[i], bot_y + bh / 2), (xs_bot[i + 1] + bw, bot_y + bh / 2),
               color=MUTED, lw=2.0)

    # The two arrows that make it a pipeline instead of a checklist.
    red = ACCENTS["red"][1]
    ax.add_patch(FancyArrowPatch(
        (xs_top[3] + 0.4, top_y + bh + 0.05), (xs_top[0] + bw / 2, top_y + bh + 0.05),
        arrowstyle="-|>", mutation_scale=16, color=red, linewidth=1.6,
        linestyle="--", connectionstyle="arc3,rad=0.22", zorder=2))
    ax.text(5.9, 8.35, "EDA keeps re-writing the question", fontsize=10.5,
            color=red, ha="center", fontstyle="italic")

    ax.add_patch(FancyArrowPatch(
        (xs_bot[2] + bw / 2, bot_y), (xs_top[1] + bw / 2, top_y),
        arrowstyle="-|>", mutation_scale=16, color=red, linewidth=1.6,
        linestyle="--", connectionstyle="arc3,rad=0.34", zorder=2))
    ax.text(4.4, 3.55, "drift -> collect again, retrain", fontsize=10,
            color=red, ha="center", fontstyle="italic")

    ax.text(0.35, 0.55,
            "Junior data scientists spend their time on stage 5. "
            "Stages 1, 3, 4 and 6 are where projects are actually won or lost.",
            fontsize=11, color=INK)
    ax.set_title("Key stages of a data science pipeline", fontsize=18,
                 fontweight="bold", color=INK, loc="left", pad=14)
    fig.tight_layout()
    return _save(fig, save_to)


# --------------------------------------------------------------------------
# Week 3: the sklearn pipeline, and cross-validation schemes
# --------------------------------------------------------------------------

def pipeline_anatomy(save_to=None):
    """What a ColumnTransformer + Pipeline actually does to your columns."""
    fig, (ax, ax_fit) = plt.subplots(
        2, 1, figsize=(15, 8.4), facecolor="white",
        gridspec_kw={"height_ratios": [3.1, 1]})

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 13.7)
    ax.axis("off")

    bl, bd = ACCENTS["blue"]
    ol, od = ACCENTS["orange"]
    gl, gd = ACCENTS["green"]
    pl, pd_ = ACCENTS["purple"]

    # --- the raw frame ----------------------------------------------------
    cols = [("distance_km", "num"), ("basket_size", "num"), ("prep_minutes", "num"),
            ("store", "cat"), ("weather", "cat"), ("courier_id", "high-card")]
    kind_colour = {"num": (bl, bd), "cat": (ol, od), "high-card": (pl, pd_)}

    _code(ax, 0.3, 13.0, "your DataFrame", fontsize=11, weight="bold")
    for i, (name, kind) in enumerate(cols):
        light, dark = kind_colour[kind]
        y = 11.1 - i * 1.35
        ax.add_patch(Rectangle((0.3, y - 0.5), 5.4, 1.0, facecolor=light,
                               edgecolor=dark, linewidth=1.3, zorder=2))
        _code(ax, 0.55, y, name, fontsize=10)

    # --- the three branches ----------------------------------------------
    # Explicit tops and heights rather than computed ones, so the boxes keep
    # a visible gap between them however many steps each branch has.
    branches = [
        ("numeric", bl, bd, ["SimpleImputer(median)", "StandardScaler()"],
         [0, 1, 2], 12.0, 3.1),
        ("categorical", ol, od, ["SimpleImputer(most_frequent)",
                                 'OneHotEncoder(', '    handle_unknown="ignore")'],
         [3, 4], 8.2, 3.9),
        ("high cardinality", pl, pd_, ["TargetEncoder()"], [5], 3.7, 2.3),
    ]
    for label, light, dark, steps, src_rows, ytop, h in branches:
        ax.add_patch(Rectangle((9.0, ytop - h), 9.2, h, facecolor=light,
                               edgecolor=dark, linewidth=1.6, zorder=2))
        _code(ax, 9.3, ytop - 0.55, label, color=dark, fontsize=10.5, weight="bold")
        for j, step in enumerate(steps):
            _code(ax, 9.5, ytop - 1.25 - j * 0.85, step, fontsize=9.5)
        for r in src_rows:
            _arrow(ax, (5.9, 11.1 - r * 1.35), (8.9, ytop - h / 2), color=dark, lw=1.3)

    _code(ax, 9.0, 13.0, "ColumnTransformer  —  one branch per kind of column",
          fontsize=11, weight="bold")
    _note(ax, 9.0, 12.4, "columns you do not list are dropped, silently", fontsize=9)

    # --- concatenate + estimator -----------------------------------------
    mid = 7.3
    ax.add_patch(Rectangle((19.5, 1.4), 2.2, 11.0, facecolor=ACCENTS["grey"][0],
                           edgecolor=ACCENTS["grey"][1], linewidth=1.4, zorder=2))
    ax.text(20.6, mid, "hstack", rotation=90, ha="center", va="center",
            fontsize=10, color=INK, **MONO)
    for _, _, _, _, _, ytop, h in branches:
        _arrow(ax, (18.3, ytop - h / 2), (19.4, mid), color=MUTED, lw=1.3)

    ax.add_patch(Rectangle((23.0, mid - 1.9), 6.5, 3.8, facecolor=gl,
                           edgecolor=gd, linewidth=1.8, zorder=2))
    _code(ax, 23.3, mid + 1.2, "estimator", color=gd, fontsize=10.5, weight="bold")
    _code(ax, 23.3, mid + 0.2, "HistGradientBoosting", fontsize=9.5)
    _code(ax, 23.3, mid - 0.6, "Regressor()", fontsize=9.5)
    _note(ax, 23.3, mid - 1.4, "or Ridge, or anything", fontsize=9)
    _arrow(ax, (21.8, mid), (22.9, mid), color=MUTED, lw=1.6)

    _code(ax, 19.5, 0.9, "Pipeline([('prep', ColumnTransformer(...)),",
          fontsize=9.5, color=MUTED)
    _code(ax, 19.5, 0.25, "          ('model', HistGradientBoostingRegressor())])",
          fontsize=9.5, color=MUTED)

    # --- why it has to be one object -------------------------------------
    ax_fit.set_xlim(0, 30)
    ax_fit.set_ylim(0, 3.4)
    ax_fit.axis("off")

    red = ACCENTS["red"][1]
    ax_fit.add_patch(Rectangle((0.3, 0.2), 29.2, 2.9, facecolor=ACCENTS["red"][0],
                               edgecolor=ACCENTS["red"][1], linewidth=1.4, zorder=1))
    _code(ax_fit, 0.8, 2.55, "Why it has to be ONE object, and not four steps in a row",
          color=red, fontsize=11, weight="bold")
    _code(ax_fit, 0.8, 1.75,
          "fit(X_train)      every step LEARNS from the training fold only   "
          "(the median, the mean and sd, the category list)",
          fontsize=9.5)
    _code(ax_fit, 0.8, 1.05,
          "predict(X_valid)  every step only APPLIES what it learned         "
          "— nothing is re-fitted on data it should not have seen",
          fontsize=9.5)
    _note(ax_fit, 0.8, 0.5,
          "Scale before you split and the validation set has already touched the model. "
          "The Pipeline is what makes that impossible to do by accident.",
          color=red, fontsize=9.5)

    fig.suptitle("Pipeline + ColumnTransformer, end to end", fontsize=18,
                 fontweight="bold", color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, save_to)


# (name, description, "use when") for each splitter drawn below
_CV_SCHEMES = [
    ("KFold(shuffle=True)", "rows shuffled, then cut into k blocks",
     "rows are independent and order does not matter", "blue"),
    ("StratifiedKFold", "same, but each fold keeps the class balance",
     "classification, especially with a rare class", "green"),
    ("GroupKFold", "whole groups go to one side or the other",
     "repeated units: same courier, user, patient, device", "purple"),
    ("TimeSeriesSplit", "always train on the past, score the future",
     "anything with a time order, or a trend", "orange"),
]


def cv_schemes(save_to=None, n_points=30, n_splits=5):
    """The four cross-validation schemes you actually need, drawn."""
    fig, axes = plt.subplots(len(_CV_SCHEMES), 1, figsize=(15, 10),
                             facecolor="white")

    rng = np.random.default_rng(0)
    # A group label per row for GroupKFold, and a time order for the rest.
    groups = np.repeat(np.arange(n_points // 3), 3)[:n_points]
    # A minority class, marked with a dot. Drawn on the first two rows so the
    # difference between KFold and StratifiedKFold is actually visible: plain
    # KFold gives each fold whatever it happens to get, stratified gives every
    # fold the same mix.
    rare = np.arange(n_points) % 3 == 0

    for ax, (name, how, when, colour) in zip(axes, _CV_SCHEMES):
        light, dark = ACCENTS[colour]
        ax.set_xlim(-17, n_points + 0.5)
        ax.set_ylim(-0.8, n_splits + 0.4)
        ax.axis("off")

        for split in range(n_splits):
            y = n_splits - 1 - split
            if name.startswith("KFold"):
                order = rng.permutation(n_points)
                val = set(order[split::n_splits])
            elif name.startswith("Stratified"):
                # Split each class separately, then combine -- which is all
                # that stratification is.
                val = set()
                for members in (np.where(rare)[0], np.where(~rare)[0]):
                    shuffled = rng.permutation(members)
                    val |= set(shuffled[split::n_splits])
            elif name.startswith("Group"):
                g_of_split = {g for g in np.unique(groups) if g % n_splits == split}
                val = {i for i in range(n_points) if groups[i] in g_of_split}
            else:  # TimeSeriesSplit: growing window, always forward
                fold = n_points // (n_splits + 1)
                train_end = fold * (split + 1)
                val = set(range(train_end, train_end + fold))

            for i in range(n_points):
                if name.startswith("TimeSeries") and i >= max(val, default=0) + 1:
                    face, edge = "white", "#e2e8f0"      # not used in this split yet
                elif i in val:
                    face, edge = dark, dark               # scored
                else:
                    face, edge = light, dark              # trained on
                ax.add_patch(Rectangle((i + 0.06, y + 0.12), 0.88, 0.76,
                                       facecolor=face, edgecolor=edge, linewidth=0.7))
                if rare[i] and name.startswith(("KFold", "Stratified")):
                    ax.plot(i + 0.5, y + 0.5, marker="o", ms=3.4,
                            color="white" if i in val else dark, zorder=3)
            label = f"split {split + 1}"
            if name.startswith(("KFold", "Stratified")):
                label += f"   {sum(rare[i] for i in val)}●"
            ax.text(-0.6, y + 0.5, label, ha="right", va="center",
                    fontsize=8.5, color=MUTED, **MONO)

        ax.text(-16.8, n_splits - 0.7, name, fontsize=12, fontweight="bold",
                color=dark, ha="left", va="center", **MONO)
        ax.text(-16.8, n_splits - 1.55, how, fontsize=9.5, color=INK,
                ha="left", va="center")
        ax.text(-16.8, n_splits - 2.3, f"use when: {when}", fontsize=9.5,
                color=MUTED, ha="left", va="center", fontstyle="italic")

        if name.startswith("Group"):
            # Show where the group boundaries fall, since that is the point.
            for i in range(1, n_points):
                if groups[i] != groups[i - 1]:
                    ax.plot([i, i], [-0.35, n_splits], color=MUTED, lw=0.7,
                            ls=":", alpha=0.8)
            ax.text(n_points / 2, -0.55, "dotted lines = group boundaries",
                    fontsize=8.5, color=MUTED, ha="center", fontstyle="italic")
        if name.startswith("Stratified"):
            ax.text(n_points / 2, -0.55,
                    "every fold is scored on the same number of minority samples",
                    fontsize=8.5, color=MUTED, ha="center", fontstyle="italic")
        if name.startswith("KFold"):
            ax.text(n_points / 2, -0.55,
                    "● = minority class; the count per fold drifts on its own",
                    fontsize=8.5, color=MUTED, ha="center", fontstyle="italic")

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=ACCENTS["blue"][0], edgecolor=ACCENTS["blue"][1]),
        Rectangle((0, 0), 1, 1, facecolor=ACCENTS["blue"][1], edgecolor=ACCENTS["blue"][1]),
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#cbd5e1"),
    ]
    fig.legend(handles, ["trained on", "scored on", "not used in this split"],
               loc="upper right", frameon=False, fontsize=10, ncol=3,
               bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Which cross-validation split?  —  one row per fold, one square per sample",
                 fontsize=17, fontweight="bold", color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, save_to)


# --------------------------------------------------------------------------
# Week 3: preprocessing, feature families, metrics
# --------------------------------------------------------------------------

def preprocessing_sheet(save_to=None):
    """What you can do to a column before it reaches a model."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.6), facecolor="white")

    blocks = [
        ("1 · missing values", "blue", [
            ("SimpleImputer(strategy=)", ['"median"   numeric, skewed', '"mean"     numeric, symmetric',
                                          '"most_frequent"  categorical', '"constant", fill_value=0']),
            ("KNNImputer()", ["borrows from similar rows", "slow, needs scaling first"]),
            ("add_indicator=True", ["keeps a was-missing flag", "often worth more than the fill"]),
        ], "Missingness is data. Before you fill, ask why it is missing."),

        ("2 · scaling", "green", [
            ("StandardScaler()", ["(x - mean) / sd", "the default"]),
            ("RobustScaler()", ["uses median and IQR", "when outliers are real"]),
            ("MinMaxScaler()", ["squashes to [0, 1]", "needs known bounds"]),
            ("PowerTransformer()", ["makes it look normal", "for skewed money-like columns"]),
        ], "Trees do not care. Linear models, SVMs and KNN care a lot."),

        ("3 · categoricals", "orange", [
            ("OneHotEncoder(", ['  handle_unknown="ignore",', '  min_frequency=20)']),
            ("OrdinalEncoder()", ["only if the order is real", "small/medium/large, not city"]),
            ("TargetEncoder()", ["high cardinality", "K-folds internally -- see nb 06"]),
        ], "Cardinality decides. Under ~15 levels one-hot; above, encode."),

        ("4 · skew and outliers", "purple", [
            ("np.log1p(x)", ["money, counts, durations", "log1p handles the zeros"]),
            ("QuantileTransformer()", ["forces a uniform/normal shape", "destroys the original units"]),
            ("winsorise: s.clip(lo, hi)", ["caps instead of dropping", "decide the caps on TRAIN only"]),
        ], "An outlier is either an error or the most important row. Decide which."),

        ("5 · dates", "red", [
            ("s.dt.hour / .dayofweek", ["never feed a raw timestamp", "a tree will split on the calendar"]),
            ("cyclical: sin/cos", ["hour 23 is next to hour 0", "two columns per cycle"]),
            ("days_since / days_until", ["holidays, launches, last event"]),
        ], "A datetime is not a feature. What you extract from it is."),

        ("6 · wiring it together", "grey", [
            ("ColumnTransformer", ["one branch per kind of column", "unlisted columns are dropped"]),
            ("Pipeline", ["preprocessing + model = one object", "fit learns, predict only applies"]),
            ("make_column_selector(", ['  dtype_include=np.number)', "picks columns by dtype"]),
        ], "If it learns anything from the data, it belongs inside the Pipeline."),
    ]

    for ax, (title, colour, entries, footer) in zip(axes.ravel(), blocks):
        light, dark = ACCENTS[colour]
        _panel(ax, title, (0, 13), (0, 11))
        y = 10.0
        for head, lines in entries:
            ax.add_patch(Rectangle((0.2, y - 0.62 - 0.72 * len(lines)), 12.6,
                                   0.75 + 0.72 * len(lines), facecolor=light,
                                   edgecolor=dark, linewidth=1.1, zorder=1))
            _code(ax, 0.45, y, head, color=dark, fontsize=9.8, weight="bold")
            for j, line in enumerate(lines):
                _code(ax, 0.75, y - 0.72 * (j + 1), line, fontsize=8.8)
            y -= 1.05 + 0.72 * len(lines)
        _note(ax, 0.2, 0.35, footer, color=dark, fontsize=9.2)

    fig.suptitle("Preprocessing: what you can do to a column",
                 fontsize=18, fontweight="bold", color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return _save(fig, save_to)


# (family, what it answers, example columns, colour)
_FEATURE_FAMILIES = [
    ("lags", "what happened N days ago?", ["lag_1", "lag_7", "lag_28"], "blue"),
    ("rolling windows", "what is the recent level?",
     ["rolling_mean_lag1_7", "rolling_std_lag1_28"], "blue"),
    ("exponential decay", "recent level, smoothly weighted", ["ewma_0.3", "ewma_0.7"], "blue"),
    ("momentum", "is it rising or falling?", ["trend_7_28", "wow_change"], "blue"),
    ("calendar", "what kind of day is it?",
     ["dow", "month", "is_holiday", "is_weekend"], "green"),
    ("cyclical encoding", "so hour 23 sits next to hour 0",
     ["dow_sin", "dow_cos", "doy_sin"], "green"),
    ("distance to events", "how close to a holiday?",
     ["days_since_holiday", "days_until_holiday"], "green"),
    ("payday / month shape", "when does money arrive?",
     ["is_payday_window", "days_to_month_end"], "green"),
    ("price and promo", "what are we charging today?",
     ["is_promo", "discount_depth", "days_since_last_promo"], "orange"),
    ("sparsity", "how often does this thing sell at all?",
     ["zero_frac_28d", "consecutive_zeros", "days_since_last_sale"], "red"),
    ("availability", "could the customer even buy it?",
     ["is_stockout", "fill_rate_7d", "availability_roll28"], "red"),
    ("target encoding", "what is this entity's own history?",
     ["variant_te_mean", "variant_te_std"], "purple"),
    ("hierarchy", "what is the parent doing?",
     ["store_rolling_mean_7d", "variant_share_of_store"], "purple"),
    ("lifecycle", "how old is this thing?",
     ["days_since_launch", "is_new_product"], "purple"),
]


def feature_families(save_to=None):
    """The families of features worth building on a panel, and what each answers."""
    # The five blocks need ~12 units between them; the rule box sits below at
    # y < 1.4, so the drawing area has to be taller than the blocks add up to.
    fig, ax = plt.subplots(figsize=(16, 11.6), facecolor="white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 14.4)
    ax.axis("off")

    groups = [("the target's own past", "blue", 0, 4),
              ("time", "green", 4, 8),
              ("what we did", "orange", 8, 9),
              ("can it even sell?", "red", 9, 11),
              ("other series", "purple", 11, 14)]

    y = 13.9
    for label, colour, lo, hi in groups:
        light, dark = ACCENTS[colour]
        block_h = 0.62 * (hi - lo) + 0.45
        ax.add_patch(Rectangle((0.2, y - block_h + 0.1), 15.6, block_h,
                               facecolor=light, edgecolor=dark, linewidth=1.3, zorder=1))
        ax.text(0.45, y - 0.12, label.upper(), fontsize=9, fontweight="bold",
                color=dark, va="top")
        for k, (name, question, cols, _) in enumerate(_FEATURE_FAMILIES[lo:hi]):
            row_y = y - 0.62 * (k + 1) - 0.05
            _code(ax, 3.2, row_y, name, fontsize=10, weight="bold")
            _note(ax, 6.5, row_y, question, fontsize=9.5, style="normal")
            _code(ax, 10.6, row_y, "  ".join(cols)[:52], color=MUTED, fontsize=8.6)
        y -= block_h + 0.25

    red = ACCENTS["red"][1]
    ax.add_patch(Rectangle((0.2, 0.15), 15.6, 1.15, facecolor=ACCENTS["red"][0],
                           edgecolor=red, linewidth=1.5, zorder=2))
    _code(ax, 0.5, 0.95, "The rule that governs every row above", color=red,
          fontsize=10.5, weight="bold")
    _note(ax, 0.5, 0.45,
          "Every one of these must be computed from data strictly before the "
          "day being predicted. In pandas that means .shift() before .rolling(), "
          "every single time.", fontsize=9.8)

    fig.suptitle("Feature families for a panel  —  one row per series per day",
                 fontsize=18, fontweight="bold", color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return _save(fig, save_to)


def metrics_sheet(save_to=None):
    """Regression metrics, what each one hides, and where the business metric lives."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.2), facecolor="white",
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    _panel(ax, "Pick the metric before you see the result", (0, 13), (0, 12.5))
    rows = [
        ("MAE", "mean |error|", "the typical miss, in real units", "blue"),
        ("RMSE", "sqrt(mean error²)", "punishes big misses; outliers dominate", "blue"),
        ("MAPE", "mean |error| / actual", "BROKEN when the actual can be 0", "red"),
        ("WMAPE", "Σ|error| / Σactual", "MAPE's fix: one ratio over the whole set", "green"),
        ("R²", "1 - SSE/SST", "share of variance explained; flatters big-variance data", "blue"),
        ("Bias", "mean(pred - actual)", "are we systematically over or under?", "orange"),
        ("Pinball", "asymmetric |error|", "for a quantile promise, not a point guess", "purple"),
    ]
    y = 11.4
    for name, formula, note, colour in rows:
        light, dark = ACCENTS[colour]
        ax.add_patch(Rectangle((0.2, y - 0.72), 12.6, 1.28, facecolor=light,
                               edgecolor=dark, linewidth=1.1, zorder=1))
        _code(ax, 0.5, y + 0.2, name, color=dark, fontsize=11, weight="bold")
        _code(ax, 2.6, y + 0.2, formula, fontsize=9.5)
        _note(ax, 0.5, y - 0.35, note, fontsize=9.3)
        y -= 1.55

    _code(ax, 0.2, 0.7, "Two questions no metric answers for you:", fontsize=10, weight="bold")
    _note(ax, 0.2, 0.15,
          "over or under -- which error costs more?    and at what grain does the "
          "decision actually get made?", fontsize=9.5)

    ax = axes[1]
    _panel(ax, "The grain the business decides at", (0, 11), (0, 12.5))
    bl, bd = ACCENTS["blue"]
    ol, od = ACCENTS["orange"]
    gl, gd = ACCENTS["green"]

    _code(ax, 0.3, 11.7, "the model predicts daily", fontsize=10.5, weight="bold")
    for i in range(10):
        ax.add_patch(Rectangle((0.3 + i * 1.02, 10.1), 0.92, 0.9,
                               facecolor=bl, edgecolor=bd, linewidth=1.1))
        ax.text(0.76 + i * 1.02, 10.55, "d", ha="center", va="center",
                fontsize=8.5, color=bd, **MONO)

    _arrow(ax, (5.4, 9.9), (5.4, 9.1), color=MUTED, lw=1.8)
    _code(ax, 0.3, 8.6, "the buyer orders every 5 days", fontsize=10.5, weight="bold")
    for i in range(2):
        ax.add_patch(Rectangle((0.3 + i * 5.1, 7.1), 5.0, 0.9,
                               facecolor=ol, edgecolor=od, linewidth=1.4))
        ax.text(2.8 + i * 5.1, 7.55, f"planning window {i + 1}", ha="center",
                va="center", fontsize=9, color=od, **MONO)

    _note(ax, 0.3, 6.4,
          "Daily errors cancel inside a window. A model that is 3 units high on\n"
          "Monday and 3 low on Tuesday is perfect for the buyer and looks\n"
          "mediocre on a daily MAE.", fontsize=9.5)

    ax.add_patch(Rectangle((0.3, 2.5), 10.4, 3.3, facecolor=gl,
                           edgecolor=gd, linewidth=1.5, zorder=1))
    _code(ax, 0.6, 5.35, "what the window makes measurable", color=gd,
          fontsize=10.5, weight="bold")
    _code(ax, 0.6, 4.6, "overstock  = max(0, predicted - actual)", fontsize=9.8)
    _note(ax, 0.6, 4.05, "stock that sat there. For fresh food, waste.", fontsize=9.2)
    _code(ax, 0.6, 3.4, "understock = max(0, actual - predicted)", fontsize=9.8)
    _note(ax, 0.6, 2.85, "orders you could not fill. Lost sale, unhappy customer.",
          fontsize=9.2)

    _note(ax, 0.3, 1.7,
          "These two are not symmetric, and the ratio between their costs is a\n"
          "business input, not something you can read off the data.", fontsize=9.5)
    _code(ax, 0.3, 0.5, "Report the metric at the grain the decision is made at.",
          color=ACCENTS["red"][1], fontsize=10, weight="bold")

    fig.suptitle("Evaluating a forecast", fontsize=18, fontweight="bold",
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, save_to)
