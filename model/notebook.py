"""Small display helpers for the three user-facing notebooks."""

from html import escape

from IPython.display import HTML, display


def show_input_tables(groups):
    """Show every input without making long tables dominate the notebook.

    The notebooks show their main settings above this expandable reference.
    No rows or columns are truncated here.
    """
    for title, table in groups.items():
        owner = str(title).split(":", 1)[0]
        source = (
            f' &middot; <a href="inputs/{owner}.toml">Edit {owner}.toml</a>'
            if owner in {"vehicle", "flight", "checks"}
            else ""
        )
        body = table.to_html(index=False, escape=True, max_rows=None, max_cols=None)
        display(
            HTML(
                f"<details><summary><strong>{escape(str(title))}</strong> "
                f"({len(table)} rows){source}</summary>{body}</details>"
            )
        )


def show_table(table):
    """Display a complete table, including long source and missing-input text."""
    display(HTML(table.to_html(index=False, escape=True, max_rows=None, max_cols=None)))
