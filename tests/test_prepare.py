"""The handoff pipeline cannot publish an incomplete or overwritten workspace."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from metso_assistant.prepare import prepare_workspace


class PreparationWorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = self.root / 'inputs'
        (self.data / 'brochures').mkdir(parents=True)
        (self.data / 'HistoricalServices').mkdir()
        (self.data / 'brochures/sample.pdf').write_bytes(b'synthetic original')
        (self.data / 'HistoricalServices/sample.txt').write_text('synthetic report', encoding='utf-8')
        self.interpreter = self.root / 'isolated-python'; self.interpreter.touch()
        self.output = self.root / 'build'
        self.calls = []

    def runner(self, command, *, cwd, stdout, stderr):
        self.calls.append(command)
        target = Path(command[command.index('--output') + 1]); target.mkdir(parents=True)
        if command[3] == 'metso_assistant.ingest':
            (target / 'manifest.json').write_text(json.dumps({'status': 'prepared', 'source_documents': 1,
                'physical_pages': 1, 'tables': 0, 'candidate_facts': 0, 'fact_review_status': {}, 'service_records': 1}), encoding='utf-8')
            (target / 'knowledge.sqlite').write_bytes(b'simulated successful child result')
        else:
            (target / 'manifest.json').write_text('{}', encoding='utf-8')
        return SimpleNamespace(returncode=0)

    def prepare(self, **options):
        return prepare_workspace(self.data, self.output, mineru_python=self.interpreter, run=self.runner, **options)

    @patch('metso_assistant.prepare.prepare_assets', return_value={'ready': True})
    def test_complete_profile_runs_both_isolated_extractors_then_services_and_knowledge(self, assets):
        before = {str(p): p.read_bytes() for p in self.data.rglob('*') if p.is_file()}
        report = self.prepare()
        self.assertEqual(report['status'], 'prepared')
        self.assertEqual([command[3] for command in self.calls], ['metso_assistant.brochure_extract',
            'metso_assistant.brochure_extract', 'metso_assistant.services', 'metso_assistant.ingest'])
        self.assertEqual(self.calls[1][0], str(self.interpreter))
        self.assertIn(str(self.data.resolve()), self.calls[3])
        self.assertEqual([s['status'] for s in report['stages']], ['complete'] * 4)
        self.assertEqual(json.loads((self.output / 'preparation.json').read_text(encoding='utf-8'))['status'], 'prepared')
        self.assertTrue(all(Path(p).read_bytes() == content for p, content in before.items()))
        assets.assert_called_once()
        self.assertFalse(assets.call_args.kwargs['download'])

    @patch('metso_assistant.prepare.prepare_assets', return_value={'ready': True})
    def test_explicit_reduced_profile_needs_no_mineru_environment(self, assets):
        self.interpreter.unlink()
        report = self.prepare(docling_only=True, download_models=True)
        self.assertEqual(report['profile'], 'docling')
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(assets.call_args.args[0], 'docling')
        self.assertTrue(assets.call_args.kwargs['download'])

    @patch('metso_assistant.prepare.prepare_assets')
    def test_existing_outputs_and_missing_inputs_fail_before_assets_or_child_execution(self, assets):
        self.output.mkdir(); sentinel = self.output / 'existing'; sentinel.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError): self.prepare()
        self.assertEqual(sentinel.read_bytes(), b'preserve')
        with self.assertRaises(FileNotFoundError):
            prepare_workspace(self.root / 'absent', self.root / 'new', run=self.runner)
        assets.assert_not_called(); self.assertEqual(self.calls, [])

    @patch('metso_assistant.prepare.prepare_assets', return_value={'ready': True})
    def test_failed_child_preserves_diagnostic_manifest_and_stops_later_stages(self, assets):
        def failure(command, **options):
            self.calls.append(command)
            options['stdout'].write(b'failure evidence')
            return SimpleNamespace(returncode=3)
        with self.assertRaisesRegex(RuntimeError, 'docling failed'):
            prepare_workspace(self.data, self.output, mineru_python=self.interpreter, run=failure)
        report = json.loads((self.output / 'preparation.json').read_text(encoding='utf-8'))
        self.assertEqual(report['status'], 'failed'); self.assertEqual(len(self.calls), 1)
        self.assertFalse((self.output / 'prepared/knowledge.sqlite').exists())
        self.assertEqual((self.output / 'logs/docling.log').read_bytes(), b'failure evidence')

    @patch('metso_assistant.prepare.prepare_assets', return_value={'ready': True})
    def test_successful_exit_without_expected_artifacts_cannot_claim_prepared(self, assets):
        with self.assertRaisesRegex(RuntimeError, 'stage docling failed'):
            prepare_workspace(self.data, self.output, mineru_python=self.interpreter,
                              run=lambda *args, **kwargs: SimpleNamespace(returncode=0))
        report = json.loads((self.output / 'preparation.json').read_text(encoding='utf-8'))
        self.assertEqual(report['status'], 'failed')

    @patch('metso_assistant.prepare.prepare_assets', return_value={'ready': True})
    def test_launch_failure_records_failed_stage(self, assets):
        def failure(*args, **kwargs):
            raise OSError('interpreter cannot start')
        with self.assertRaises(OSError):
            prepare_workspace(self.data, self.output, mineru_python=self.interpreter, run=failure)
        report = json.loads((self.output / 'preparation.json').read_text(encoding='utf-8'))
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['stages'][0]['status'], 'failed')

    @patch('metso_assistant.prepare.prepare_assets', return_value={'ready': True})
    def test_child_manifest_without_valid_store_is_not_a_complete_handoff(self, assets):
        for invalid in ('status', 'missing_store'):
            self.output = self.root / invalid
            def incomplete(command, **options):
                result = self.runner(command, **options)
                if command[3] == 'metso_assistant.ingest':
                    prepared = self.output / 'prepared'
                    if invalid == 'status':
                        manifest = prepared / 'manifest.json'
                        value = json.loads(manifest.read_text(encoding='utf-8')); value['status'] = 'failed'
                        manifest.write_text(json.dumps(value), encoding='utf-8')
                    else:
                        (prepared / 'knowledge.sqlite').unlink()
                return result
            with self.subTest(invalid=invalid), self.assertRaisesRegex(RuntimeError, 'complete knowledge store'):
                prepare_workspace(self.data, self.output, mineru_python=self.interpreter, run=incomplete)
            report = json.loads((self.output / 'preparation.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
