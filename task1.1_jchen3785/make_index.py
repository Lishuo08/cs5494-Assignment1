"""Generate index.html: a one-page map of the deliverable for the team.

Run `python make_index.py` after adding or removing files; the page lists every
file with its size and a one-line description, plus the key numbers and the
things that are still open.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'index.html')

# path -> description shown in the table
FILES = {
    'START_HERE.md': 'reading order, quickstart, 中文摘要 - start with this one',
    'README.md': 'environment, run order, data contract, figures, all numbers',
    'readme.txt': 'the same run instructions in plain text (assignment asks for readme.txt)',
    'requirements.txt': 'pinned package versions used for the numbers',
    'HANDOFF.md': 'what is done, key numbers, what I still need from tasks 1.2 / 1.3',
    'message_to_teammates.txt': 'ready-to-paste message for B and C (Chinese): links, numbering, what I need',
    'index.html': 'this page',

    'data.py': 'shared MNIST loader, binarisation, pixel order, subset, plotting',
    'fvsbn.py': 'the model: logits(), nll(), ancestral sample()',
    'train_fvsbn.py': 'one training run (Adam, staged samples, timings, JSON/npz/pt)',
    'run_experiments.py': 'main run + 3 ablations (13 runs in total)',
    'make_figures.py': 'loss curve, stage grid, sample grid, ablation figures',
    'analyze_fvsbn.py': 'unconditional baseline, pixel accuracy, predictive maps',
    'summarize.py': 'all runs -> build/results_summary.csv / .md',
    'build_demo_nb.py': 'builds and executes demo_data.ipynb',
    'demo_data.ipynb': 'executed tour of the shared data module (11 cells, 5 code)',
    'build_section_pdf.py': 'renders the report section to PDF',
    'make_index.py': 'builds this page',
    'check_package.py': 'verifies that the folder and the zip are complete',
    'pack_zip.py': 'rebuilds the zip (forward slashes, so it unzips cleanly on any OS)',

    'report/section_1_1_fvsbn.md': 'my report section, Markdown source',
    'report/section_1_1_fvsbn.pdf': 'the same section rendered, 5 pages with figures',
    'report/report_skeleton.md': 'team report outline; TODOs mark the 1.2 / 1.3 parts',

    'build/results_summary.csv': 'every run: config, params, NLL, bits/dim, timings',
    'build/results_summary.md': 'the same table as Markdown',
    'build/analysis.json': 'baseline, pixel accuracy, ink fractions',
    'build/figs/fig_loss.png': 'Figure 3.1 training / test curves',
    'build/figs/fig_stages.png': 'Figure 3.2 samples at epochs 1, 5, 10, 25, 50',
    'build/figs/fig_samples.png': 'Figure 3.3 40 final samples',
    'build/figs/fig_predmap.png': 'Figure 3.4 test image, conditional probability, argmax',
    'build/figs/fig_ablation.png': 'Figure 3.5 ablation bars',
    'build/figs/fig_ablation_samples.png': 'samples for every ablation setting',
    'build/figs/fig_pixel_acc.png': 'per-pixel teacher-forcing accuracy map',
}

GROUPS = [
    ('Read first', ['START_HERE.md', 'index.html', 'HANDOFF.md',
                    'message_to_teammates.txt', 'README.md', 'readme.txt',
                    'requirements.txt']),
    ('Report', ['report/section_1_1_fvsbn.pdf', 'report/section_1_1_fvsbn.md',
                'report/report_skeleton.md']),
    ('Code', ['data.py', 'fvsbn.py', 'train_fvsbn.py', 'run_experiments.py',
              'make_figures.py', 'analyze_fvsbn.py', 'summarize.py',
              'build_demo_nb.py', 'demo_data.ipynb']),
    ('Results and figures', ['build/results_summary.csv', 'build/results_summary.md',
                             'build/analysis.json', 'build/figs/fig_loss.png',
                             'build/figs/fig_stages.png', 'build/figs/fig_samples.png',
                             'build/figs/fig_predmap.png', 'build/figs/fig_ablation.png',
                             'build/figs/fig_ablation_samples.png',
                             'build/figs/fig_pixel_acc.png']),
    ('Housekeeping', ['build_section_pdf.py', 'make_index.py', 'check_package.py',
                      'pack_zip.py']),
]

NUMBERS = [
    ('unconditional Bernoulli', '205.85', '0.3788', '-', '-', '-'),
    ('FVSBN, 60k images, 50 epochs', '75.75', '<b>0.1394</b>', '615,440', '30-50 s', '0.06 s'),
]


def size(path):
    if path == 'index.html':
        return 'this page'
    full = os.path.join(HERE, path.replace('/', os.sep))
    if not os.path.exists(full):
        return '<span class="missing">missing</span>'
    n = os.path.getsize(full)
    if n < 1024:
        return '%d B' % n
    if n < 1024 * 1024:
        return '%.0f KB' % (n / 1024.0)
    return '%.1f MB' % (n / 1048576.0)


def rows():
    out = []
    for title, paths in GROUPS:
        out.append('<tr class="group"><td colspan="3">%s</td></tr>' % title)
        for p in paths:
            out.append('<tr><td><a href="%s">%s</a></td><td class="size">%s</td>'
                       '<td>%s</td></tr>' % (p, p, size(p), FILES.get(p, '')))
    return '\n'.join(out)


HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<title>CS5494 Assignment 1 - task 1.1 package</title>
<meta name="author" content="CHEN Jiaqi (jchen3785)">
<style>
body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; font-size: 10.5pt;
       color: #1f1f1f; max-width: 62rem; margin: 2rem auto; padding: 0 1.5rem; }}
h1 {{ font-size: 1.5rem; margin: 0 0 .2rem 0; }}
.sub {{ color: #555; margin-bottom: 1.2rem; }}
h2 {{ font-size: 1.05rem; margin: 1.6rem 0 .5rem 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: .92rem; }}
th, td {{ border-bottom: 1px solid #ddd; padding: .35rem .5rem; text-align: left;
          vertical-align: top; }}
th {{ background: #f4f4f4; }}
tr.group td {{ background: #faeff2; font-weight: 600; color: #8c1d40; }}
.size {{ white-space: nowrap; color: #666; width: 6rem; }}
.missing {{ color: #b00020; }}
code {{ background: #f5f5f5; padding: .1rem .3rem; }}
.box {{ background: #f7f9fb; border-left: 4px solid #8c1d40; padding: .7rem 1rem;
        margin: 1rem 0; font-size: .93rem; }}
ol {{ margin: .3rem 0 .3rem 1.2rem; }}
</style></head>
<body>
<h1>CS5494 Assignment 1 &mdash; task 1.1: Fully Visible Sigmoid Belief Network</h1>
<div class="sub">CHEN Jiaqi &nbsp;|&nbsp; jchen3785 &nbsp;|&nbsp; CityU MSc AI &nbsp;|&nbsp;
merged report due Sunday 8 November 2026</div>

<div class="box">
<b>Quickstart.</b> <code>pip install -r requirements.txt</code> &rarr;
<code>python run_experiments.py</code> (~2-3 min on the GPU) &rarr;
<code>python make_figures.py</code>. The report section is
<a href="report/section_1_1_fvsbn.pdf">report/section_1_1_fvsbn.pdf</a>.
</div>

<h2>What is in this folder</h2>
<table>
<tr><th>file</th><th>size</th><th>what it is</th></tr>
{rows}
</table>

<h2>Key numbers (test split, binarised at threshold 0.5)</h2>
<table>
<tr><th>model</th><th>NLL / image</th><th>bits / dim</th><th>params</th>
<th>training</th><th>100 samples</th></tr>
{numbers}
</table>
<p>Bits per dimension is NLL / (784 * ln 2); lower is better. The unconditional
baseline predicts each pixel from its training marginal and ignores the prefix.</p>

<h2>Still open</h2>
<ol>
<li>Task 1.2 (neural autoregressive) and task 1.3 (GAN) sections, plus their rows in the
comparison table of <a href="report/report_skeleton.md">report/report_skeleton.md</a>.</li>
<li>The shared code link (Colab / Drive) for the merged report.</li>
<li>The final file must be named <code>&lt;id&gt;_&lt;id&gt;_&lt;id&gt;.pdf</code>.</li>
<li>All three tasks must use <code>data.load_binarized</code> (1.2) and
<code>data.load_gray</code> (1.3), otherwise the bits/dim column is not comparable.</li>
</ol>

<p class="sub">Regenerate this page with <code>python make_index.py</code>; check the folder
with <code>python check_package.py</code>.</p>
</body></html>
"""


def main():
    nums = '\n'.join('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
                     '<td>%s</td></tr>' % n for n in NUMBERS)
    html = HTML.format(rows=rows(), numbers=nums)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(OUT, len(html), 'chars')


if __name__ == '__main__':
    main()
