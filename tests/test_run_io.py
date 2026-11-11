"""Small filesystem-only regressions for notebook output safeguards."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.run_io import (mock_output_bundle, load_mock_metadata,
                          prepare_sampler_run, prepare_figure_output)


class OutputSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prefix = self.root / "mock"

    def bundle(self, **kwargs):
        with mock_output_bundle(self.prefix, **kwargs) as products:
            products['npz'].write_bytes(b'mock bytes')
            products['metadata'].write_text(json.dumps({'ddr': {'model': 'M1'}}))
            products['diagnostic'].write_bytes(b'figure bytes')

    def test_bundle_records_checksum_and_refuses_overwrite(self):
        self.bundle()
        self.assertIn('dataset_sha256', load_mock_metadata(self.prefix))
        with self.assertRaises(FileExistsError):
            self.bundle()
        self.assertFalse(Path(str(self.prefix) + '.write.lock').exists())

    def test_failed_staging_preserves_previous_bundle(self):
        self.bundle()
        before = Path(str(self.prefix) + '_paired_dataset.npz').read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'diagnostic'):
            with mock_output_bundle(self.prefix, overwrite=True) as products:
                products['npz'].write_bytes(b'replacement')
                raise RuntimeError('diagnostic failed')
        self.assertEqual(Path(str(self.prefix) + '_paired_dataset.npz').read_bytes(), before)

    def test_incomplete_staging_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'All NPZ'):
            with mock_output_bundle(self.prefix) as products:
                products['npz'].write_bytes(b'incomplete')
        self.assertFalse(Path(str(self.prefix) + '_paired_dataset.npz').exists())

    def test_checksum_and_unfinished_lock_are_rejected(self):
        self.bundle()
        Path(str(self.prefix) + '_paired_dataset.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_mock_metadata(self.prefix)
        Path(str(self.prefix) + '.write.lock').touch()
        with self.assertRaisesRegex(RuntimeError, 'lock'):
            load_mock_metadata(self.prefix)

    def test_legacy_metadata_remains_readable(self):
        Path(str(self.prefix) + '_metadata.json').write_text('{"legacy": true}')
        self.assertTrue(load_mock_metadata(self.prefix)['legacy'])

    def info(self):
        return {'output': str(self.root/'chains'/'test'),
                'params': {'alpha': {'prior': {'min': -0.9, 'max': 1.5}}}}

    def files(self):
        data=self.root/'data.npz'; data.write_bytes(b'dataset')
        meta=self.root/'data.json'; meta.write_text('{}')
        return data, meta

    def test_fresh_and_matching_resume_are_safe(self):
        data, meta=self.files(); info=self.info()
        fresh=prepare_sampler_run(info, data, meta)
        self.assertFalse(fresh['force']); self.assertFalse(fresh['resume'])
        Path(info['output'] + '.1.txt').write_text('chain')
        resumed=prepare_sampler_run(info, data, meta, resume=True)
        self.assertTrue(resumed['resume']); self.assertFalse(resumed['force'])
        with self.assertRaises(FileExistsError):
            prepare_sampler_run(info, data, meta)

    def test_resume_rejects_changed_inputs_and_configuration(self):
        data, meta=self.files(); info=self.info()
        prepare_sampler_run(info, data, meta)
        Path(info['output'] + '.1.txt').write_text('chain')
        data.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            prepare_sampler_run(info, data, meta, resume=True)
        data.write_bytes(b'dataset'); info['params']['alpha']['prior']['max']=2
        with self.assertRaisesRegex(ValueError, 'changed'):
            prepare_sampler_run(info, data, meta, resume=True)

    def test_resume_and_overwrite_cannot_be_combined(self):
        data, meta=self.files()
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            prepare_sampler_run(self.info(), data, meta, resume=True, overwrite=True)

    def test_legacy_chain_cannot_be_resumed_without_identity(self):
        data, meta=self.files(); info=self.info()
        Path(info['output']).parent.mkdir()
        Path(info['output'] + '.1.txt').touch()
        with self.assertRaises(FileNotFoundError):
            prepare_sampler_run(info, data, meta, resume=True)
        self.assertTrue(prepare_sampler_run(info, data, meta, overwrite=True)['force'])

    def test_figure_directory_and_overwrite_guard(self):
        path=self.root/'figures'/'comparison.pdf'
        self.assertEqual(prepare_figure_output(path),path)
        self.assertTrue(path.parent.is_dir()); path.touch()
        with self.assertRaises(FileExistsError):
            prepare_figure_output(path)
        self.assertEqual(prepare_figure_output(path,overwrite=True),path)
