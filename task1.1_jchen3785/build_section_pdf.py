"""Render report/section_1_1_fvsbn.md to a PDF with the figures inline.

Markdown is the source of truth. Images in the markdown are relative paths into
build/figs; they are embedded as base64 so the HTML is self-contained, and Edge
prints it headlessly to report/section_1_1_fvsbn.pdf.
"""

import base64
import os
import re
import subprocess
import tempfile

import markdown
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, 'report', 'section_1_1_fvsbn.md')
HTML = os.path.join(HERE, 'build', 'section_1_1.html')
PDF = os.path.join(HERE, 'report', 'section_1_1_fvsbn.pdf')

EDGE = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'

CSS = """
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.5;
       margin: 0 auto; max-width: 16.5cm; color: #1a1a1a; }
h1 { font-size: 15pt; margin: 0 0 2px 0; }
h2 { font-size: 12pt; margin: 18px 0 6px 0; page-break-after: avoid; }
p { margin: 0 0 9px 0; text-align: justify; }
code { font-family: Consolas, monospace; font-size: 9pt; }
pre { font-family: Georgia, serif; font-size: 10pt; margin: 4px 0 10px 22px;
      white-space: pre-wrap; }
table { border-collapse: collapse; margin: 10px 0 14px 0; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3px 8px; }
th { background: #efefef; }
figure { margin: 10px 0 16px 0; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 12.5cm; }
figcaption { font-size: 9pt; color: #333; margin-top: 4px; text-align: center; }
.byline { font-size: 9.5pt; color: #444; margin-bottom: 2px; }
.toc { font-size: 9pt; color: #666; margin-bottom: 14px; }
"""


def embed(path):
    im = Image.open(path).convert('RGB')
    gray = im.convert('L').point(lambda v: 0 if v > 246 else 255)
    box = gray.getbbox()
    if box:
        pad = 10
        box = (max(0, box[0] - pad), max(0, box[1] - pad),
               min(im.width, box[2] + pad), min(im.height, box[3] + pad))
        im = im.crop(box)
    out = path.replace('\\', '/').replace('/', '_')
    tmp = os.path.join(HERE, 'build', 'embedded_' + os.path.basename(out))
    im.save(tmp)
    with open(tmp, 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode('ascii')


def figures(html):
    pattern = re.compile(r'<p><img alt="([^"]*)" src="([^"]+)"\s*/?></p>', re.S)

    def repl(m):
        caption, src = m.group(1), m.group(2)
        abspath = os.path.normpath(os.path.join(HERE, 'report', src))
        return ('<figure><img src="%s"><figcaption>%s</figcaption></figure>'
                % (embed(abspath), caption))

    return pattern.sub(repl, html)


def main():
    with open(MD, encoding='utf-8') as f:
        text = f.read()
    body = figures(markdown.markdown(text, extensions=['tables', 'fenced_code']))
    byline = ('<div class="byline">CHEN Jiaqi &nbsp;|&nbsp; jchen3785 &nbsp;|&nbsp; '
              'CS5494 Assignment 1, draft section</div>'
              '<div class="toc">3.1 Model &nbsp;&middot;&nbsp; 3.2 Experimental setup '
              '&nbsp;&middot;&nbsp; 3.3 Results &nbsp;&middot;&nbsp; 3.4 Discussion '
              '&nbsp;&middot;&nbsp; 3.5 Limitations</div>')
    body = body.replace('</h1>', '</h1>' + byline, 1)

    html = ('<!doctype html><html><head><meta charset="utf-8">'
            '<title>CS5494 Assignment 1 - Task 1.1 FVSBN (CHEN Jiaqi, jchen3785)</title>'
            '<meta name="author" content="CHEN Jiaqi">'
            '<meta name="description" content="Report section for task 1.1: fully visible '
            'sigmoid belief network on binarised MNIST">'
            '<style>%s</style></head><body>%s</body></html>' % (CSS, body))
    with open(HTML, 'w', encoding='utf-8') as f:
        f.write(html)

    tmp = tempfile.mkdtemp(prefix='edge_sec_')
    subprocess.run([EDGE, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                    '--user-data-dir=' + tmp, '--print-to-pdf=' + PDF,
                    'file:///' + HTML.replace('\\', '/')], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(PDF, 'rb') as f:
        data = f.read()
    print(PDF, len(data), 'pages', data.count(b'/Type /Page') - data.count(b'/Type /Pages'))


if __name__ == '__main__':
    main()
