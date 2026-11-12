"""Persistence and HTTP contract checks; no simulator dependencies required."""
import copy
import io
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
import zipfile
from unittest.mock import patch
import urllib.error
import urllib.request

from server import Handler, Platform, SCENARIOS, VERSION


class PlatformChecks(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        models = []
        for scene in SCENARIOS:
            checkpoint = self.root / (scene + '.pt')
            checkpoint.write_bytes(b'test fixture')
            models.append(dict(id=scene, label=scene, scenario=scene, algorithm='test', checkpoint=str(checkpoint), config=None))
        catalog = self.root / 'catalog.json'
        catalog.write_text(json.dumps(dict(models=models)), encoding='utf-8')
        self.app = Platform(catalog, self.root / 'runs')

    def result(self, index=1, **updates):
        sid = f'{index:032x}'
        folder = self.app.output / sid
        folder.mkdir(exist_ok=True)
        item = dict(id=sid, model={k:v for k,v in self.app.models['warehouse'].items() if k not in {'checkpoint','config'}},
                    seed=101, status='completed', started_at='2026-11-04T00:00:00+00:00',
                    finished_at='2026-11-04T00:00:07+00:00', error=None,
                    frame_count=25, frame=dict(step=24, metrics=[dict(label='利润', value=12, unit='')]))
        item.update(updates)
        (folder/'result.json').write_text(json.dumps(item, ensure_ascii=False), encoding='utf-8')
        os.utime(folder/'result.json', (1000+index, 1000+index))
        return folder, item

    def start_http(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        server.app = self.app
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(2)
        self.addCleanup(cleanup)
        return f'http://127.0.0.1:{server.server_port}'

    def test_history_exposes_summary_and_preserves_old_keys(self):
        _, item = self.result()
        row = self.app.history()[0]
        for key in ('id','model','seed','status','started_at'):
            self.assertEqual(row[key], item[key])
        self.assertEqual(row['frame_count'], 25)
        self.assertEqual(row['last_step'], 24)
        self.assertEqual(row['metrics'], item['frame']['metrics'])
        self.assertEqual(row['duration_seconds'], 7)

    def test_legacy_history_counts_trace_without_assuming_step_equals_frames(self):
        folder, item = self.result()
        del item['frame_count']
        (folder/'result.json').write_text(json.dumps(item), encoding='utf-8')
        (folder/'frames.jsonl').write_text('{}\n{}\n\n{}\n', encoding='utf-8')
        self.assertEqual(self.app.history()[0]['frame_count'], 3)

    def test_failed_legacy_run_without_trace_remains_visible(self):
        folder, item = self.result(status='failed', frame=None, error='model load failed')
        del item['frame_count']
        (folder/'result.json').write_text(json.dumps(item), encoding='utf-8')
        row = self.app.history()[0]
        self.assertEqual(row['frame_count'], 0)
        self.assertEqual(row['error'], 'model load failed')
        self.assertIsNone(row['last_step'])

    def test_corrupt_recent_results_do_not_hide_older_valid_runs(self):
        self.result(1)
        for index in range(2, 23):
            folder, _ = self.result(index)
            (folder/'result.json').write_text('{incomplete', encoding='utf-8')
        self.assertEqual([row['id'] for row in self.app.history()], [f'{1:032x}'])

    def test_malformed_structures_and_nonfinite_values_are_skipped(self):
        self.result(1)
        self.result(2, model=[])
        self.result(3, frame={'metrics':[{'value':float('nan')}]})
        self.result(4, id='../../bad')
        self.result(5, status='ready')
        self.assertEqual([row['id'] for row in self.app.history()], [f'{1:032x}'])

    def test_health_reports_missing_assets_without_absolute_paths(self):
        healthy = self.app.health()
        self.assertEqual(healthy['status'], 'ok')
        self.assertEqual(healthy['version'], VERSION)
        self.assertEqual(healthy['device'], 'cpu')
        self.assertEqual(healthy['current'], {'status':'idle'})
        Path(self.app.models['traffic']['checkpoint']).unlink()
        degraded = self.app.health()
        self.assertEqual(degraded['status'], 'degraded')
        self.assertFalse(degraded['scenarios']['traffic'])
        payload = json.dumps(degraded)
        self.assertNotIn(str(self.root), payload)
        self.assertNotIn('checkpoint', payload)

    def test_http_health_and_host_restriction(self):
        url = self.start_http()
        with urllib.request.urlopen(url+'/api/health') as response:
            self.assertEqual(json.load(response)['execution'], 'local')
        request = urllib.request.Request(url+'/api/health', headers={'Host':'untrusted.invalid'})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 403)

    def test_parameters_are_validated_before_a_worker_or_run_is_created(self):
        for values in ({'rows':3, 'cols':2, 'controller':'trained'},
                       {'rows':True}, {'vehicle_count':-1},
                       {'vehicle_count':3001}, {'decision_limit':181},
                       {'controller':'shell'}, {'unknown':'value'}):
            with self.subTest(parameters=values), patch('server.Session') as session:
                with self.assertRaises(ValueError):
                    self.app.start({'model':'traffic', 'seed':101, 'parameters':values})
                session.assert_not_called()
        self.assertEqual(list(self.app.output.iterdir()), [])

    def test_pressure_run_preserves_source_model_and_labels_actual_controller(self):
        parameters = dict(rows=3, cols=2, vehicle_count=100, decision_limit=8, controller='pressure')
        with patch('server.Session.run', return_value=None):
            state = self.app.start({'model':'traffic', 'seed':202, 'parameters':parameters})
            self.app.session.thread.join(2)
        self.assertEqual(state['parameters'], parameters)
        self.assertEqual(state['effective_controller'], 'pressure')
        self.assertEqual(state['model']['algorithm'], 'rule/pressure')
        self.assertEqual(state['source_model']['algorithm'], 'test')
        request = json.loads((self.app.session.folder/'request.json').read_text(encoding='utf-8'))
        self.assertEqual(request['parameters'], parameters)
        self.assertEqual(request['effective_algorithm'], 'rule/pressure')
        self.assertEqual(request['model']['algorithm'], 'test')
        self.assertEqual(self.app.health()['current']['algorithm'], 'rule/pressure')

    def test_legacy_default_run_keeps_empty_parameters_and_trained_identity(self):
        with patch('server.Session.run', return_value=None):
            state = self.app.start({'model':'warehouse'})
            self.app.session.thread.join(2)
        self.assertEqual(state['parameters'], {})
        self.assertEqual(state['effective_controller'], 'trained')
        self.assertEqual(state['model']['algorithm'], 'test')

    def test_pressure_history_does_not_mislabel_legacy_model_metadata(self):
        model = {k:v for k,v in self.app.models['traffic'].items() if k not in {'checkpoint','config'}}
        self.result(model=model, parameters={'controller':'pressure', 'rows':3, 'cols':3})
        row = self.app.history()[0]
        self.assertEqual(row['model']['algorithm'], 'rule/pressure')
        self.assertEqual(row['source_model']['algorithm'], 'test')

    def test_custom_parameters_for_other_scenes_are_rejected_before_start(self):
        for scene in ('warehouse', 'mpe', 'powergrid'):
            with self.subTest(scene=scene), patch('server.Session') as session:
                with self.assertRaises(ValueError):
                    self.app.start({'model':scene, 'parameters':{'vehicle_count':10}})
                session.assert_not_called()

    def test_http_invalid_parameters_return_400_without_starting_simulator(self):
        url = self.start_http()
        request = urllib.request.Request(url+'/api/start', data=json.dumps(dict(model='traffic', parameters={'rows':4,'controller':'trained'})).encode(),
                                         headers={'Content-Type':'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 400)
        self.assertIn('error', json.load(error.exception))
        self.assertIsNone(self.app.session)

    def test_http_scenario_schema_exposes_traffic_defaults(self):
        url = self.start_http()
        with urllib.request.urlopen(url+'/api/scenarios') as response:
            data = json.load(response)
        self.assertEqual(data['traffic']['trained_grid'], [2,2])
        self.assertEqual(data['traffic']['defaults']['controller'], 'trained')

    def test_corrupt_export_returns_json_422_and_server_stays_usable(self):
        folder, item = self.result()
        (folder/'request.json').write_text('{}', encoding='utf-8')
        (folder/'frames.jsonl').write_text('{incomplete', encoding='utf-8')
        url = self.start_http()
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url+'/api/export/'+item['id'])
        self.assertEqual(error.exception.code, 422)
        self.assertIn('error', json.load(error.exception))
        with urllib.request.urlopen(url+'/api/history') as response:
            self.assertEqual(len(json.load(response)), 1)

    def test_http_training_export_has_assets_and_never_starts_a_worker(self):
        from scenarios import HERE
        import yaml
        self.app.models['traffic']['config'] = str(HERE / 'bundle/configs/traffic.yaml')
        url = self.start_http()
        data = dict(model='traffic', algorithm='pymarl/vdn', seed=101,
                    parameters=dict(rows=3, cols=1, vehicle_count=19, decision_limit=8, controller='trained'))
        request = urllib.request.Request(url+'/api/training/export', data=json.dumps(data).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.headers.get_content_type(), 'application/zip')
            with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                experiment = yaml.safe_load(archive.read('experiment.yaml'))
                self.assertEqual(experiment['algorithm'], 'pymarl/vdn')
                self.assertEqual(len(json.loads(archive.read('scenario/flow.json'))), 19)
        self.assertIsNone(self.app.session)
        self.assertEqual(list(self.app.output.iterdir()), [])

    def test_registered_traffic_rejects_scene_replacement_but_allows_rule(self):
        self.app.models['traffic']['inference_protocol'] = 'generic_traffic_v1'
        with patch('server.Session') as session:
            with self.assertRaises(ValueError):
                self.app.start(dict(model='traffic', parameters={'rows': 2}))
            session.assert_not_called()
        with patch('server.Session.run', return_value=None):
            state = self.app.start(dict(model='traffic', parameters={'controller': 'pressure'}))
            self.app.session.thread.join(2)
        self.assertEqual(state['model']['algorithm'], 'rule/pressure')
        self.assertEqual(state['parameters'], {'controller': 'pressure'})

    def test_http_compare_rejects_bad_inputs_without_crashing_server(self):
        url = self.start_http()
        request = urllib.request.Request(url+'/api/compare', data=json.dumps({'ids': ['../bad', 'a'*32]}).encode(),
                                         headers={'Content-Type': 'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 400)
        self.assertIn('error', json.load(error.exception))


if __name__ == '__main__':
    unittest.main()
