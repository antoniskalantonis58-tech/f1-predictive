#!/usr/bin/env python3
"""
Convert Markdown report to HTML with professional styling for PDF export.
Usage: python convert_report_to_html.py
"""

import re
import markdown
from pathlib import Path

def convert_markdown_to_html(md_file, html_file):
    """Convert Markdown to HTML with custom CSS for PDF printing."""
    
    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Clean file paths for PDF export
    md_content = re.sub(r'`[^`]*f1 predictive/[^`]*`', '`main.py`', md_content)
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'])
    body_html = md.convert(md_content)
    
    # Create full HTML with CSS
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>F1 Predictive Modeling Report</title>
    <style>
        /* A4 page setup */
        @page {{
            size: A4;
            margin: 2.5cm 2cm;
        }}
        
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: "Times New Roman", Times, serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #000;
            max-width: 21cm;
            margin: 0 auto;
            padding: 2cm;
            background: white;
        }}
        
        /* Headings */
        h1 {{
            font-size: 20pt;
            font-weight: bold;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            page-break-after: avoid;
            text-align: center;
        }}
        
        h2 {{
            font-size: 16pt;
            font-weight: bold;
            margin-top: 1.2em;
            margin-bottom: 0.5em;
            page-break-after: avoid;
            border-bottom: 1px solid #333;
            padding-bottom: 0.3em;
        }}
        
        h3 {{
            font-size: 13pt;
            font-weight: bold;
            margin-top: 1em;
            margin-bottom: 0.4em;
            page-break-after: avoid;
        }}
        
        /* Paragraphs */
        p {{
            margin: 0.5em 0;
            text-align: justify;
            orphans: 3;
            widows: 3;
        }}
        
        /* Links */
        a {{
            color: #0066cc;
            text-decoration: none;
        }}
        
        @media print {{
            a {{
                color: #000;
            }}
        }}
        
        /* Code and technical elements */
        code {{
            font-family: "Courier New", Courier, monospace;
            font-size: 10pt;
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        
        pre {{
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            overflow-x: auto;
            font-size: 9pt;
            line-height: 1.4;
        }}
        
        pre code {{
            background: none;
            padding: 0;
        }}
        
        /* Tables */
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
            font-size: 10pt;
        }}
        
        th, td {{
            border: 1px solid #000;
            padding: 8px;
            text-align: left;
        }}
        
        th {{
            background-color: #f0f0f0;
            font-weight: bold;
        }}
        
        td:nth-child(2), td:nth-child(3) {{
            text-align: right;
        }}
        
        /* Lists */
        ul, ol {{
            margin: 0.5em 0;
            padding-left: 2em;
        }}
        
        li {{
            margin: 0.3em 0;
        }}
        
        /* Blockquotes */
        blockquote {{
            margin: 1em 2em;
            padding-left: 1em;
            border-left: 3px solid #ccc;
            font-style: italic;
        }}
        
        /* Print optimizations */
        @media print {{
            body {{
                padding: 0;
                max-width: 100%;
            }}
            
            h1, h2, h3 {{
                page-break-after: avoid;
            }}
            
            table, figure, img {{
                page-break-inside: avoid;
            }}
            
            pre, blockquote {{
                page-break-inside: avoid;
            }}
            
            /* Hide file paths in print */
            .file-path {{
                display: none;
            }}
        }}
        
        /* Horizontal rules */
        hr {{
            border: none;
            border-top: 1px solid #ccc;
            margin: 2em 0;
        }}
        
        /* Abstract and keywords */
        .abstract {{
            background-color: #f9f9f9;
            padding: 1em;
            margin: 1em 0;
            border-left: 4px solid #333;
        }}
        
        .keywords {{
            font-style: italic;
            margin-top: 0.5em;
        }}
        
        /* Page breaks */
        .page-break {{
            page-break-before: always;
        }}
        
        /* Header info */
        .header-info {{
            text-align: center;
            margin-bottom: 2em;
        }}
        
        .header-info p {{
            margin: 0.2em 0;
            text-align: center;
        }}
    </style>
</head>
<body>
    {body_html}
    
    <script>
        // Clean up table of contents for PDF
        document.addEventListener('DOMContentLoaded', function() {{
            // Remove any remaining Markdown link syntax
            document.body.innerHTML = document.body.innerHTML
                .replace(/\\[([^\\]]+)\\]\\(#[^)]+\\)/g, '$1');
        }});
    </script>
</body>
</html>"""
    
    # Write HTML file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"✅ HTML file created: {html_file}")
    print(f"📄 File paths have been cleaned for PDF export")
    print(f"\n📋 To create PDF:")
    print(f"   1. Open {html_file} in your browser")
    print(f"   2. Press Cmd+P (Print)")
    print(f"   3. Select 'Save as PDF'")
    print(f"   4. Verify it's approximately 10 pages")

if __name__ == "__main__":
    md_file = Path("project_report_final.md")
    html_file = Path("project_report_final.html")
    
    if not md_file.exists():
        print(f"❌ Error: {md_file} not found!")
        exit(1)
    
    convert_markdown_to_html(md_file, html_file)
