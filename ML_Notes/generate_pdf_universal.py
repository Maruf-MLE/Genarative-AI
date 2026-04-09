#!/usr/bin/env python3
"""
Universal Markdown → HTML → PDF Converter
Light Theme — পাঠযোগ্য, সুন্দর ফরম্যাট।
Usage: python generate_pdf_universal.py <markdown_file_path>
"""

import os
import re
import sys
import subprocess

# ─────────────────────────────────────────────────────────────
# Beautiful Light-Themed CSS (Linear Regression style)
# ─────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Hind Siliguri', 'Inter', 'Segoe UI', sans-serif;
    background-color: #ffffff;
    color: #1a1a2e;
    line-height: 1.85;
    font-size: 15.5px;
    padding: 0;
}

.container {
    max-width: 900px;
    margin: 0 auto;
    padding: 50px 60px;
}

/* ── Top Banner ── */
.doc-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 50%, #1a8fc1 100%);
    color: white;
    padding: 40px 50px;
    border-radius: 14px;
    margin-bottom: 45px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(30, 58, 95, 0.25);
}
.doc-banner::after {
    content: '';
    position: absolute;
    bottom: -30px; right: -30px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
}
.doc-banner h1 {
    font-size: 1.75em;
    font-weight: 700;
    color: white !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 0 10px 0 !important;
    background: none !important;
    -webkit-text-fill-color: white !important;
    letter-spacing: -0.3px;
}
.doc-banner p {
    color: rgba(255,255,255,0.85);
    font-size: 0.95em;
    margin: 4px 0;
}

/* ── Headings ── */
h1 {
    font-size: 1.75em;
    font-weight: 700;
    color: #1e3a5f;
    margin: 45px 0 18px;
    padding-bottom: 10px;
    border-bottom: 3px solid #2d6a9f;
    letter-spacing: -0.3px;
}

h2 {
    font-size: 1.35em;
    font-weight: 600;
    color: #1e5799;
    margin: 35px 0 14px;
    padding: 12px 18px;
    background: linear-gradient(135deg, #eef4ff, #f0f7ff);
    border-left: 5px solid #2d6a9f;
    border-radius: 0 8px 8px 0;
    box-shadow: inset 0 0 0 1px rgba(45,106,159,0.1);
}

h3 {
    font-size: 1.15em;
    font-weight: 600;
    color: #1a5276;
    margin: 25px 0 11px;
    padding-left: 12px;
    border-left: 3px solid #5dade2;
}

h4 {
    font-size: 1.05em;
    font-weight: 600;
    color: #2471a3;
    margin: 18px 0 9px;
}

/* ── Paragraphs ── */
p {
    margin: 11px 0;
    color: #2c3e50;
}

/* ── Code Blocks ── */
pre {
    background: #f4f6f9;
    border: 1px solid #d5e3f0;
    border-left: 5px solid #2d6a9f;
    border-radius: 8px;
    padding: 20px 22px;
    margin: 18px 0;
    overflow-x: auto;
    font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.75;
    color: #1a1a2e;
    page-break-inside: avoid;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

code {
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    background: #e8f0fa;
    color: #1a5276;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.87em;
    border: 1px solid #c5d8ef;
}

pre code {
    background: none;
    color: inherit;
    padding: 0;
    border: none;
    font-size: inherit;
    border-radius: 0;
}

/* ── Tables ── */
table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin: 20px 0;
    border-radius: 10px;
    overflow: hidden;
    font-size: 0.93em;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    page-break-inside: avoid;
}

th {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
    color: #ffffff;
    padding: 13px 18px;
    text-align: left;
    font-weight: 600;
    font-size: 0.95em;
    letter-spacing: 0.3px;
}

td {
    padding: 11px 18px;
    border-bottom: 1px solid #e2ecf5;
    color: #2c3e50;
    background: #ffffff;
}

tr:nth-child(even) td {
    background: #f5f9fe;
}

tr:last-child td {
    border-bottom: none;
}

/* ── Lists ── */
ul, ol {
    margin: 12px 0 12px 22px;
    padding: 0;
}

li {
    margin: 7px 0;
    color: #2c3e50;
    padding-left: 4px;
}

ul li::marker {
    color: #2d6a9f;
    font-size: 1.1em;
}

ol li::marker {
    color: #1e5799;
    font-weight: 700;
}

/* ── Blockquote ── */
blockquote {
    border-left: 5px solid #2d6a9f;
    background: linear-gradient(135deg, #eef4ff, #f5f9ff);
    padding: 14px 20px;
    margin: 18px 0;
    border-radius: 0 8px 8px 0;
    color: #1a3a5c;
    font-style: italic;
    box-shadow: 0 2px 8px rgba(45,106,159,0.08);
}

/* ── Horizontal Rule ── */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, #2d6a9f, #5dade2, transparent);
    margin: 35px 0;
    border-radius: 2px;
}

/* ── Links ── */
a {
    color: #1a6fc4;
    text-decoration: none;
}

/* ── Strong / Em ── */
strong {
    color: #1a3a5c;
    font-weight: 700;
}

em {
    color: #1a5276;
    font-style: italic;
}

/* ── Section Cards (for numbered sections) ── */
.section-card {
    background: #ffffff;
    border: 1px solid #e2ecf5;
    border-radius: 12px;
    padding: 28px 32px;
    margin: 24px 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    page-break-inside: avoid;
}

/* ── Footer ── */
.doc-footer {
    margin-top: 55px;
    padding: 22px 30px;
    background: linear-gradient(135deg, #eef4ff, #f5f9ff);
    border-radius: 10px;
    text-align: center;
    color: #5d7ea4;
    font-size: 0.88em;
    border: 1px solid #d5e3f0;
}

/* ── Print Optimization ── */
@media print {
    body {
        background: #ffffff !important;
        color: #1a1a2e !important;
        font-size: 13px;
    }
    .container {
        max-width: 100%;
        padding: 20px 30px;
    }
    h1 { color: #1e3a5f !important; font-size: 1.5em; }
    h2 { 
        color: #1e5799 !important; 
        font-size: 1.2em;
        background: #eef4ff !important;
        border-left: 5px solid #2d6a9f !important;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }
    h3 { color: #1a5276 !important; font-size: 1.05em; }
    pre {
        background: #f4f6f9 !important;
        border: 1px solid #ccc !important;
        border-left: 5px solid #2d6a9f !important;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
        page-break-inside: avoid;
    }
    table { page-break-inside: avoid; }
    th {
        background: #1e3a5f !important;
        color: #ffffff !important;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }
    tr:nth-child(even) td {
        background: #f0f5fb !important;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }
    blockquote {
        background: #eef4ff !important;
        border-left: 5px solid #2d6a9f !important;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }
    .doc-banner {
        background: #1e3a5f !important;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }
    .doc-footer {
        background: #eef4ff !important;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }
}
"""

# ─────────────────────────────────────────────────────────────
# Markdown → HTML Converter
# ─────────────────────────────────────────────────────────────
def md_to_html(md_text):
    html = md_text

    # Escape HTML special chars in non-code sections first (handle separately)
    # Code blocks — protect first
    code_blocks = {}
    counter = [0]

    def protect_code(m):
        key = f"CODEBLOCK{counter[0]}CODEBLOCK"
        lang = m.group(1).strip() if m.group(1) else ""
        content = m.group(2)
        # Simple syntax: just show as-is
        escaped = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        code_blocks[key] = f'<pre><code class="lang-{lang}">{escaped}</code></pre>'
        counter[0] += 1
        return key
    
    html = re.sub(r'```(\w*)\n?(.*?)```', protect_code, html, flags=re.DOTALL)

    # Inline code — protect
    inline_codes = {}
    ic_counter = [0]
    def protect_inline(m):
        key = f"INLINECODE{ic_counter[0]}INLINECODE"
        escaped = m.group(1).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        inline_codes[key] = f'<code>{escaped}</code>'
        ic_counter[0] += 1
        return key
    html = re.sub(r'`([^`\n]+)`', protect_inline, html)

    # Horizontal rule
    html = re.sub(r'^---+$', '<hr>', html, flags=re.MULTILINE)

    # Headings
    html = re.sub(r'^###### (.+)$', r'<h6>\1</h6>', html, flags=re.MULTILINE)
    html = re.sub(r'^##### (.+)$',  r'<h5>\1</h5>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.+)$',   r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$',    r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$',     r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$',      r'<h1>\1</h1>', html, flags=re.MULTILINE)

    # Bold + Italic
    html = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html)
    html = re.sub(r'\*\*(.+?)\*\*',     r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*',         r'<em>\1</em>', html)

    # Blockquote
    html = re.sub(r'^> (.+)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)

    # Tables
    def process_table(match):
        table_text = match.group(0).strip()
        lines = [l.strip() for l in table_text.split('\n') if l.strip()]
        if len(lines) < 2:
            return match.group(0)
        result = '<table>\n<thead>\n'
        for i, line in enumerate(lines):
            if re.match(r'^\|[-:| ]+\|$', line):
                result += '</thead>\n<tbody>\n'
                continue
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if i == 0:
                result += '<tr>' + ''.join(f'<th>{c}</th>' for c in cells) + '</tr>\n'
            else:
                result += '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>\n'
        result += '</tbody>\n</table>'
        return result

    html = re.sub(r'(\|.+\|\n?)+', process_table, html)

    # Unordered lists
    lines = html.split('\n')
    result_lines = []
    in_ul = False
    for line in lines:
        if re.match(r'^[-*] .+', line):
            if not in_ul:
                result_lines.append('<ul>')
                in_ul = True
            item = re.sub(r'^[-*] ', '', line)
            result_lines.append(f'  <li>{item}</li>')
        else:
            if in_ul:
                result_lines.append('</ul>')
                in_ul = False
            result_lines.append(line)
    if in_ul:
        result_lines.append('</ul>')
    html = '\n'.join(result_lines)

    # Ordered lists
    lines = html.split('\n')
    result_lines = []
    in_ol = False
    for line in lines:
        if re.match(r'^\d+\. .+', line):
            if not in_ol:
                result_lines.append('<ol>')
                in_ol = True
            item = re.sub(r'^\d+\. ', '', line)
            result_lines.append(f'  <li>{item}</li>')
        else:
            if in_ol:
                result_lines.append('</ol>')
                in_ol = False
            result_lines.append(line)
    if in_ol:
        result_lines.append('</ol>')
    html = '\n'.join(result_lines)

    # Links
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)

    # Paragraphs
    lines = html.split('\n')
    result_lines = []
    block_tags = ('CODEBLOCK', 'INLINECODE', '<h', '<ul', '<ol', '<li', '<table',
                  '<thead', '<tbody', '<tr', '<th', '<td', '<blockquote', '<hr',
                  '</ul', '</ol', '</table', '</thead', '</tbody', '</h')
    for line in lines:
        stripped = line.strip()
        if stripped and not any(stripped.startswith(t) for t in block_tags):
            result_lines.append(f'<p>{line}</p>')
        else:
            result_lines.append(line)
    html = '\n'.join(result_lines)

    # Restore protected blocks
    for key, val in code_blocks.items():
        html = html.replace(f'<p>{key}</p>', val)
        html = html.replace(key, val)
    for key, val in inline_codes.items():
        html = html.replace(key, val)

    return html


# ─────────────────────────────────────────────────────────────
# Full HTML Template
# ─────────────────────────────────────────────────────────────
def build_full_html(title, body_html, subtitle=""):
    return f"""<!DOCTYPE html>
<html lang="bn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{CSS}</style>
</head>
<body>
<div class="container">

  <div class="doc-banner">
    <h1>📚 {title}</h1>
    <p>{subtitle}</p>
    <p>🗓️ তৈরির তারিখ: ২০২৬-০৪-০৪ &nbsp;|&nbsp; সম্পূর্ণ বাংলায় লেখা ML Notes</p>
  </div>

{body_html}

  <div class="doc-footer">
    <p>📚 Machine Learning &amp; Deep Learning Notes — সম্পূর্ণ বাংলায়</p>
    <p>🤖 AI-assisted Bengali ML Notes | D:\\Coding\\Genarative-AI\\ML_Notes</p>
  </div>

</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
# Edge PDF Generator
# ─────────────────────────────────────────────────────────────
def generate_pdf(md_path):
    md_path = os.path.abspath(md_path)
    base    = os.path.splitext(md_path)[0]
    html_path = base + "_temp.html"
    pdf_path  = base + ".pdf"
    title     = os.path.basename(base).replace('_', ' ')

    print(f"\n{'='*55}")
    print(f"📄 Processing: {os.path.basename(md_path)}")
    print(f"{'='*55}")

    # Read markdown
    print("  📖 Markdown পড়া হচ্ছে...")
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Convert
    print("  🔄 HTML-এ রূপান্তর হচ্ছে...")
    body = md_to_html(md_content)
    full_html = build_full_html(title, body, "বিস্তারিত বাংলা ML নোট | Intuition · Math · Code · Diagrams")

    # Save HTML
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"  💾 HTML সেভ হয়েছে ({os.path.getsize(html_path)//1024} KB)")

    # Find Edge
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    edge_exe = next((p for p in edge_paths if os.path.exists(p)), None)

    if not edge_exe:
        print("  ❌ Microsoft Edge পাওয়া যায়নি!")
        return False

    # Generate PDF
    print(f"  🖨️  PDF তৈরি হচ্ছে...")
    cmd = [
        edge_exe,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--run-all-compositor-stages-before-draw",
        "--disable-extensions",
        f"--print-to-pdf={pdf_path}",
        "--print-to-pdf-no-header",
        html_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        print("  ⚠️ Edge timeout হয়েছে, আবার চেষ্টা করা হচ্ছে...")
        result = subprocess.run(cmd[:], capture_output=True, text=True, timeout=90)

    # Check result
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 5000:
        size_kb = os.path.getsize(pdf_path) // 1024
        print(f"  ✅ PDF তৈরি সফল! ({size_kb} KB)")
        print(f"  📄 Path: {pdf_path}")
        # Cleanup HTML
        os.remove(html_path)
        print(f"  🗑️  Temp HTML মুছে ফেলা হয়েছে।")
        return True
    else:
        print(f"  ❌ PDF তৈরি ব্যর্থ হয়েছে।")
        if result.stderr:
            print(f"  Error: {result.stderr[:200]}")
        print(f"  💡 HTML ফাইলটি রয়ে গেছে: {html_path}")
        return False


# ─────────────────────────────────────────────────────────────
# Main — দুটো ফাইলের জন্য একসাথে চালাও
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # নির্দিষ্ট ফাইলগুলো
    files_to_convert = [
        r"D:\Coding\Genarative-AI\ML_Notes\Logistic_Regression\Logistic_Regression.md",
        r"D:\Coding\Genarative-AI\ML_Notes\ANN_Introduction\ANN_Introduction.md",
    ]

    # Command line argument দিলে সেটা ব্যবহার করো
    if len(sys.argv) > 1:
        files_to_convert = sys.argv[1:]

    results = {}
    for md_file in files_to_convert:
        if not os.path.exists(md_file):
            print(f"⚠️ ফাইল পাওয়া যায়নি: {md_file}")
            results[md_file] = False
            continue
        results[md_file] = generate_pdf(md_file)

    # Summary
    print(f"\n{'='*55}")
    print("📊 সারসংক্ষেপ:")
    print(f"{'='*55}")
    for f, ok in results.items():
        status = "✅ সফল" if ok else "❌ ব্যর্থ"
        print(f"  {status}  →  {os.path.basename(f)}")
    print()
