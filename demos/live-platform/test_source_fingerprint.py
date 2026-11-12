"""Regression checks for detecting changed downloads without forcing network use."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from source_fingerprint import INPUTS, fingerprint


class SourceFingerprintChecks(unittest.TestCase):
    def fixture(self, root):
        for name in INPUTS:
            path = root / name
            if name in ('pyproject.toml', 'README.md', 'LICENSE', 'NOTICE'):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('same source', encoding='utf-8')
            else:
                path.mkdir(parents=True, exist_ok=True)
                (path / 'input.txt').write_text('same source', encoding='utf-8')

    def test_moving_identical_download_keeps_fingerprint(self):
        with TemporaryDirectory() as directory:
            a, b = Path(directory) / 'download one', Path(directory) / 'new location'
            self.fixture(a)
            self.fixture(b)
            self.assertEqual(fingerprint(a), fingerprint(b))

    def test_web_source_changes_and_new_model_trigger_rebuild(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            before = fingerprint(root)
            web = root / 'demos/live-platform/static/app.js'
            web.parent.mkdir(parents=True)
            web.write_text('updated UI', encoding='utf-8')
            changed = fingerprint(root)
            self.assertNotEqual(before, changed)
            model = root / 'demos/live-platform/bundle/models/new/policy.pt'
            model.parent.mkdir(parents=True)
            model.write_bytes(b'new model')
            self.assertNotEqual(changed, fingerprint(root))

    def test_run_records_and_build_caches_do_not_trigger_rebuild(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            before = fingerprint(root)
            for relative in ('demos/live-platform/local-state/runs/id/frames.jsonl',
                             'src/__pycache__/test.pyc',
                             'demos/live-platform/bundle/vendor/CityFlow/build/output.so',
                             'demos/live-platform/bundle/vendor/CityFlow/CityFlow.egg-info/PKG-INFO'):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'runtime data')
            self.assertEqual(before, fingerprint(root))


if __name__ == '__main__':
    unittest.main()
