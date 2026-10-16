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
           'studio_review_auth', 'studio_review_tasks', 'studio_review_server', 'studio_collaboration', 'crop_policy')
PATHS = {name: ROOT/'tools'/(name+'.py') for name in MODULES if name != 'crop_policy'}
PATHS['crop_policy'] = ROOT/'crop_policy.py'
FILES = {os.path.normcase(str(path)): name for name, path in PATHS.items()}
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
    suite = unittest.TestSuite([unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern)
                               for pattern in ['test_studio*.py', 'test_final_crop_policy.py']])
    result = unittest.TextTestRunner().run(suite)
    sys.settrace(None)
    threading.settrace(None)
    rows = []
    for name in MODULES:
        lines = set(trace._find_executable_linenos(str(PATHS[name])))
        rows.append({'module': name, 'covered': len(lines & executed[name]), 'executable': len(lines)})
    covered, total = sum(row['covered'] for row in rows), sum(row['executable'] for row in rows)
    report = {'scope': 'eleven Python backend/policy modules; excludes research core, frontend and C# host',
              'tests': result.testsRun, 'passed': result.wasSuccessful(),
              'covered': covered, 'executable': total, 'percent': round(100*covered/total, 1), 'modules': rows}
    output = ROOT/'data/desktop_build/session_backend_coverage.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    sys.exit(0 if result.wasSuccessful() and covered/total >= .8 else 1)
