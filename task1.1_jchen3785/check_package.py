"""Check that the handoff folder (and the zip, if present) is complete.

Prints one line per expected file with its size, then a summary: missing files,
files that are not valid UTF-8, PDF page count, zip entry count and zip CRC
check. Exit code is 0 when everything is fine, 1 otherwise.
"""

import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, 'CS5494_task1.1_jchen3785.zip')

EXPECTED = [
    'START_HERE.md', 'README.md', 'readme.txt', 'requirements.txt', 'HANDOFF.md',
    'message_to_teammates.txt', 'index.html',
    'data.py', 'fvsbn.py', 'train_fvsbn.py', 'run_experiments.py',
    'make_figures.py', 'analyze_fvsbn.py', 'summarize.py', 'build_demo_nb.py',
    'demo_data.ipynb', 'build_section_pdf.py', 'make_index.py', 'check_package.py',
    'pack_zip.py',
    'report/section_1_1_fvsbn.md', 'report/section_1_1_fvsbn.pdf',
    'report/report_skeleton.md', 'build/results_summary.csv',
    'build/results_summary.md', 'build/analysis.json',
]

FIGURES = ['fig_loss', 'fig_stages', 'fig_samples', 'fig_predmap',
           'fig_ablation', 'fig_ablation_samples', 'fig_pixel_acc']


def check_files():
    missing = []
    for rel in EXPECTED:
        path = os.path.join(HERE, rel.replace('/', os.sep))
        if os.path.exists(path):
            print('  %-42s %9d' % (rel, os.path.getsize(path)))
        else:
            print('  %-42s MISSING' % rel)
            missing.append(rel)
    for name in FIGURES:
        rel = 'build/figs/%s.png' % name
        path = os.path.join(HERE, rel.replace('/', os.sep))
        if os.path.exists(path):
            print('  %-42s %9d' % (rel, os.path.getsize(path)))
        else:
            print('  %-42s MISSING' % rel)
            missing.append(rel)
    return missing


def check_text():
    bad = []
    for rel in EXPECTED:
        if not rel.endswith(('.md', '.txt', '.py', '.csv', '.json')):
            continue
        path = os.path.join(HERE, rel.replace('/', os.sep))
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            data = f.read()
        try:
            data.decode('utf-8')
        except UnicodeDecodeError as exc:
            print('  not UTF-8:', rel, exc)
            bad.append(rel)
    return bad


def check_pdf():
    path = os.path.join(HERE, 'report', 'section_1_1_fvsbn.pdf')
    if not os.path.exists(path):
        return None
    data = open(path, 'rb').read()
    return data.count(b'/Type /Page') - data.count(b'/Type /Pages')


def check_zip():
    if not os.path.exists(ZIP):
        return None, 0, 'no zip'
    zf = zipfile.ZipFile(ZIP)
    bad = zf.testzip()
    count = len(zf.namelist())
    status = 'crc ok' if bad is None else 'corrupt: ' + str(bad)
    return zf, count, status


def main():
    print('folder:', HERE)
    print('files:')
    missing = check_files()
    bad = check_text()
    pages = check_pdf()

    print('\nsummary')
    print('  missing files      :', len(missing))
    print('  non UTF-8 files    :', len(bad))
    print('  section PDF pages  :', pages)

    zf, count, status = check_zip()
    if count:
        print('  zip entries        :', count, '(', status, ')')
    else:
        print('  zip                : not present (normal inside an extracted copy)')
    if zf is not None:
        names = zf.namelist()
        zf.close()
        expected = {e.replace('/', os.sep) for e in EXPECTED}
        expected = {e.replace(os.sep, '/') for e in expected}
        extra = [n for n in names if n not in expected and not n.startswith('build/figs/')]
        if extra:
            print('  zip contains files not listed here:', extra)

    ok = not missing and not bad and bool(pages)
    if count and not status.startswith('crc ok'):
        ok = False
    print('\nRESULT:', 'OK' if ok else 'CHECK FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
