import os
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'CS5494_task1.1_jchen3785.zip')

FILES = [
    'START_HERE.md', 'README.md', 'readme.txt', 'requirements.txt', 'HANDOFF.md',
    'message_to_teammates.txt', 'index.html',
    'data.py', 'fvsbn.py', 'train_fvsbn.py', 'run_experiments.py', 'make_figures.py',
    'analyze_fvsbn.py', 'summarize.py', 'build_demo_nb.py', 'demo_data.ipynb',
    'build_section_pdf.py', 'make_index.py', 'check_package.py', 'pack_zip.py',
    'report/section_1_1_fvsbn.md', 'report/section_1_1_fvsbn.pdf',
    'report/report_skeleton.md',
    'build/results_summary.csv', 'build/results_summary.md', 'build/analysis.json',
]
FIG_DIR = os.path.join(ROOT, 'build', 'figs')
FILES += ['build/figs/' + n for n in sorted(os.listdir(FIG_DIR)) if n.endswith('.png')]


def main():
    if os.path.exists(OUT):
        os.remove(OUT)
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
        for rel in FILES:
            path = os.path.join(ROOT, rel.replace('/', os.sep))
            zf.write(path, rel)
    with zipfile.ZipFile(OUT) as zf:
        names = zf.namelist()
        bad = [n for n in names if '\\' in n]
        print('zip', OUT)
        print('entries', len(names), 'forward slashes', not bad)
        print('testzip', zf.testzip())
        print('size', os.path.getsize(OUT))


if __name__ == '__main__':
    main()
