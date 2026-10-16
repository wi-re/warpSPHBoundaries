"""Portable HTML companion and explicit figure links for Markdown reports."""
import html
import re
from pathlib import Path


def accessible_report(path):
    """Update display files only; never rerun evaluators or modify cached data."""
    path = Path(path)
    text = path.read_text()
    text = re.sub(r'(?m)^!\[([^\]]*)\]\(([^)]+)\)(?!\n\n\[Open figure)',
                  lambda m: m[0]+f'\n\n[Open figure — {m[1]}]({m[2]})', text)
    text = text.replace('[Browser report with inline figures](report.html)', '[Optional HTML companion](report.html)')
    link = '[Optional HTML companion](report.html)'
    if link not in text:
        title, rest = text.split('\n', 1)
        text = title+'\n\n'+link+' · Figure links below also open images individually.\n'+rest
    path.write_text(text)
    try:
        from markdown_it import MarkdownIt
    except ImportError as exc:
        raise RuntimeError('HTML report needs the scripts extra: pip install -e ".[scripts]"') from exc
    body = MarkdownIt('commonmark', {'html': False}).enable('table').render(text)
    # Preserve the known generated section anchors without enabling arbitrary HTML.
    body = re.sub(r'&lt;a id=&quot;([a-z0-9-]+)&quot;&gt;&lt;/a&gt;', r'<a id="\1"></a>', body)
    title = html.escape(next((line[2:] for line in text.splitlines() if line.startswith('# ')), 'Boundary audit'))
    path.with_suffix('.html').write_text(f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>
body {{font:17px/1.55 system-ui,sans-serif;color:#19242e;max-width:1250px;margin:2em auto;padding:0 1.2em}}
img {{display:block;max-width:100%;height:auto;margin:1em auto;background:white}}
a {{color:#075aa8}} table {{border-collapse:collapse;display:block;overflow:auto}}
th,td {{padding:.5em .7em;border:1px solid #cbd3da;text-align:left}}
pre {{padding:1em;background:#f0f3f6;overflow:auto}} h2 {{margin-top:2.5em}}
@media print {{body {{max-width:none}} img, tr {{break-inside:avoid}}}}
</style><body>{body}</body></html>''')
    return path.with_suffix('.html')
