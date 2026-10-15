"""Measure executable Python backend lines exercised by studio unit/integration tests."""
import json
import os
import sys
import threading
import trace
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('studio_project', 'studio_service', 'studio_predictor', 'studio_lifecycle', 'crop_studio', 'studio_session',
           'studio_review_auth', 'studio_review_tasks', 'studio_review_server', 'studio_collaboration')
FILES = {os.path.normcase(str(ROOT/'tools'/(name+'.py'))): name for name in MODULES}
executed = {name: set() for name in MODULES}


def count_lines(frame, event, arg):
    name = FILES.get(os.path.normcase(frame.f_code.co_filename))
    if name is None:
        return None
    if event == 'line':
        executed[name].add(frame.f_lineno)
    return count_lines


if __name__ == '__main__':
    sys.settrace(count_lines)
    threading.settrace(count_lines)
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.discover(str(ROOT/'tests'), 'test_studio*.py'))
    sys.settrace(None)
    threading.settrace(None)
    rows = []
    for name in MODULES:
        lines = set(trace._find_executable_linenos(str(ROOT/'tools'/(name+'.py'))))
        rows.append({'module': name, 'covered': len(lines & executed[name]), 'executable': len(lines)})
    covered, total = sum(row['covered'] for row in rows), sum(row['executable'] for row in rows)
    report = {'scope': 'ten Python studio and collaboration backend modules; excludes frontend and C# host',
              'tests': result.testsRun, 'passed': result.wasSuccessful(),
              'covered': covered, 'executable': total, 'percent': round(100*covered/total, 1), 'modules': rows}
    output = ROOT/'data/desktop_build/session_backend_coverage.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    sys.exit(0 if result.wasSuccessful() and covered/total >= .8 else 1)
