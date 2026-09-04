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
