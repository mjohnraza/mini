import os
import re

def markdown_to_html(md_path: str, html_path: str):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Basic markdown parsing for titles, tables, code blocks
    html_lines = []
    in_code = False
    in_table = False

    template_head = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MiniGPT Technical Report - Transformer & GPT From Scratch</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body);"></script>
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #24292e; }
    h1 { font-size: 2.2em; border-bottom: 2px solid #eaecef; padding-bottom: 0.3em; }
    h2 { font-size: 1.6em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-top: 24px; color: #1a5276; }
    h3 { font-size: 1.25em; margin-top: 20px; color: #2c3e50; }
    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
    th, td { border: 1px solid #dfe2e5; padding: 8px 14px; text-align: left; }
    th { background-color: #f6f8fa; font-weight: 600; }
    code { background-color: #f0f3f6; padding: 0.2em 0.4em; border-radius: 3px; font-family: SFMono-Regular, Consolas, monospace; font-size: 85%; }
    pre code { display: block; padding: 12px; overflow-x: auto; background-color: #282c34; color: #abb2bf; border-radius: 6px; }
    blockquote { border-left: 4px solid #3498db; margin: 0; padding-left: 16px; color: #555; }
    img { max-width: 100%; height: auto; border: 1px solid #e1e4e8; border-radius: 6px; margin: 15px 0; }
</style>
</head>
<body>
"""

    for line in content.splitlines():
        if line.startswith("```"):
            if in_code:
                html_lines.append("</code></pre>")
                in_code = False
            else:
                html_lines.append("<pre><code>")
                in_code = True
            continue
        if in_code:
            html_lines.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # Tables
        if "|" in line:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if all(set(p).issubset({"-", ":"}) for p in parts if p):
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                html_lines.append("<table><thead><tr>" + "".join(f"<th>{p}</th>" for p in parts) + "</tr></thead><tbody>")
                in_table = True
            else:
                html_lines.append("<tr>" + "".join(f"<td>{p}</td>" for p in parts) + "</tr>")
            continue
        elif in_table:
            html_lines.append("</tbody></table>")
            in_table = False

        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("- "):
            html_lines.append(f"<li>{line[2:]}</li>")
        elif line.strip() == "---":
            html_lines.append("<hr>")
        elif line.strip():
            # Bold & italic formatting
            txt = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            txt = re.sub(r'`(.*?)`', r'<code>\1</code>', txt)
            html_lines.append(f"<p>{txt}</p>")

    if in_table:
        html_lines.append("</tbody></table>")

    template_foot = """
</body>
</html>
"""
    full_html = template_head + "\n".join(html_lines) + template_foot
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"[HTML Report] Compiled styled report to: {html_path}")

if __name__ == "__main__":
    markdown_to_html("report/report.md", "report/report.html")
