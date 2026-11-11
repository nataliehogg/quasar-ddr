"""Tests for input validation and notebook configuration."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.qso_mock import generate_paired_log_mock, load_paired_log_mock, save_paired_log_mock
from tools.QSO_specifications import QSOSpecs
from tools.ddr_likelihood import predict_log_eta
from tools.error_models import resolve_sarm_error, resolve_uvx_error
from tools.theory import angular_diameter_distance, ddr_eta
from tools.run_io import configuration_id, mock_output_bundle, load_mock_metadata
from tools.utilities import Utilities
from tests.test_likelihood import load_likelihood_module

ROOT=Path(__file__).resolve().parents[1]


def cell(name,index):
    return ''.join(json.loads((ROOT/name).read_text())['cells'][index]['source'])


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.mock=generate_paired_log_mock(np.array([0.2,0.8]),np.array([300.,700.]),0.925,0.3,0.2)

    def test_bad_arrays_are_rejected(self):
        for field,values in [('dl_observed',[0,1]),('da_observed',[np.nan,1]),
                             ('sigma_ln_dl',[-1,0]),('sigma_ln_da',[np.inf,0]),
                             ('rho',[2,0]),('redshift',[-1,0]),('da_true',[1])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(self.mock,**{field:np.asarray(values)})

    def test_corrupt_npz_is_rejected_at_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.npz'
            fields=dict(self.mock.__dict__); fields['seed']=-1
            fields['dl_observed']=np.array([np.nan,1])
            np.savez(path,**fields)
            with self.assertRaisesRegex(ValueError,'finite'):
                load_paired_log_mock(path)
            fields['dl_observed']=self.mock.dl_observed;fields['seed']=1.5
            np.savez(path,**fields)
            with self.assertRaisesRegex(ValueError,'seed'):
                load_paired_log_mock(path)

    def test_zero_total_variance_fails_but_one_zero_component_is_valid(self):
        module=load_likelihood_module()
        with tempfile.TemporaryDirectory() as directory:
            prefix=str(Path(directory)/'mock')
            for dl,da,valid in [(0.,0.,False),(0.,0.2,True),(0.3,0.,True)]:
                mock=generate_paired_log_mock(self.mock.redshift,self.mock.da_true,0.925,dl,da)
                save_paired_log_mock(prefix+'_paired_dataset.npz',mock)
                like=module.DDRlike(); like.datapath=prefix; like.model='M1'
                if valid:
                    like.initialize(); self.assertAlmostEqual(like.logp(alpha=-0.075),0.)
                    self.assertEqual(like.logp(alpha=np.nan),-np.inf)
                else:
                    with self.assertRaisesRegex(ValueError,'positive total'):
                        like.initialize()

    def test_nonfinite_predictions_and_specifications_fail(self):
        with self.assertRaises(ValueError):predict_log_eta([0.2], 'M1',alpha=np.nan)
        with self.assertRaises(ValueError):predict_log_eta([np.nan], 'M2')
        for specs in [{'redshifts':[0.2,np.nan]}, {'z_min':0.2,'z_max':2.,'n_QSO':1.5},
                      {'z_min':0.2,'z_max':np.inf,'n_QSO':2}]:
            with self.subTest(specs=specs), self.assertRaises(ValueError):
                QSOSpecs.redshift_distribution(specs,mode='asimov')


class NotebookPlumbingTests(unittest.TestCase):
    def namespace(self):
        ns={'np':np,'Path':Path,'json':json,'configuration_id':configuration_id,
            'QSOSpecs':QSOSpecs,'angular_diameter_distance':angular_diameter_distance,
            'ddr_eta':ddr_eta,'resolve_sarm_error':resolve_sarm_error,
            'resolve_uvx_error':resolve_uvx_error,'generate_paired_log_mock':generate_paired_log_mock}
        for i in (4,6,8):exec(cell('mock_maker_interface.ipynb',i),ns)
        ns['specs']['n_QSO']=3
        return ns

    def test_custom_errors_and_seed_are_forwarded_without_changing_streams(self):
        ns=self.namespace()
        ns['errors'].update(uvx='custom',sarm='custom',custom_sigma_ln_dl=0.4,custom_sigma_ln_da=0.15)
        ns['mode']='random'
        exec(cell('mock_maker_interface.ipynb',12),ns)
        mock=ns['mock']; self.assertEqual(mock.seed,27)
        np.testing.assert_allclose(mock.sigma_ln_dl,0.4)
        np.testing.assert_allclose(mock.sigma_ln_da,0.15)
        children=np.random.SeedSequence(27).spawn(3)
        expected=generate_paired_log_mock(ns['z'],ns['da'],ns['eta'],0.4,0.15,
                                         mode='random',seed=27,rng=np.random.default_rng(children[1]))
        np.testing.assert_array_equal(mock.dl_observed,expected.dl_observed)

    def test_paper_priors_and_convergence_overrides_are_preserved(self):
        from tools.run_io import prepare_sampler_run
        module=load_likelihood_module()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); data=root/'mock_paired_dataset.npz'; data.write_bytes(b'fixture')
            prefix=root/'mock'; Path(str(prefix)+'_metadata.json').write_text('{}')
            class LocalUtilities:
                @staticmethod
                def sampler_config(dataset, likelihood):
                    info=Utilities.sampler_config(dataset,likelihood)
                    info['output']=str(root/'chains'/'paper')
                    return info
            ns={'Utilities':LocalUtilities,'DDRlike':module.DDRlike,'Path':Path,
                'prepare_sampler_run':prepare_sampler_run,'mock_file':data,'prefix':prefix,
                'dataset':{'model':'M2','alpha':-0.07,'beta':-0.014,'settings':'mock'}}
            exec(cell('sampler_interface.ipynb',6),ns)
            info=ns['info']
            self.assertEqual(info['params']['alpha']['prior'],{'min':-0.9,'max':1.5})
            self.assertEqual(info['params']['beta']['prior'],{'min':-1.,'max':1.})
            self.assertEqual(info['sampler']['mcmc']['Rminus1_stop'],0.01)
            self.assertEqual(info['sampler']['mcmc']['Rminus1_cl_stop'],0.05)
            self.assertFalse(info['force']); self.assertFalse(info['resume'])

    def test_default_paper_paths_are_retained_and_alternatives_isolated(self):
        ns=self.namespace();exec(cell('mock_maker_interface.ipynb',10),ns)
        self.assertEqual(ns['outdir'],Path('mock_data'))
        ns['specs']['n_QSO']=3.0
        exec(cell('mock_maker_interface.ipynb',10),ns)
        self.assertIn('nQSO3_current',ns['stem'])
        ns['seed']=28;exec(cell('mock_maker_interface.ipynb',10),ns)
        self.assertIn('asimov_seed28',str(ns['outdir']))
        first=ns['outdir']; ns['cosmo']['H0']=70.
        exec(cell('mock_maker_interface.ipynb',10),ns)
        self.assertNotEqual(first,ns['outdir'])

    def test_actual_notebook_save_cell_round_trip(self):
        from datetime import datetime,timezone
        from tools.mockchecker import save_paired_mock_diagnostic
        ns=self.namespace();exec(cell('mock_maker_interface.ipynb',12),ns)
        ns.update(datetime=datetime,timezone=timezone,mock_output_bundle=mock_output_bundle,
                  save_paired_log_mock=save_paired_log_mock,save_paired_mock_diagnostic=save_paired_mock_diagnostic)
        with tempfile.TemporaryDirectory() as directory:
            ns.update(outdir=Path(directory),stem='tiny')
            exec(cell('mock_maker_interface.ipynb',14),ns)
            meta=load_mock_metadata(Path(directory)/'tiny')
            loaded=load_paired_log_mock(Path(directory)/'tiny_paired_dataset.npz')
            self.assertEqual(meta['random_seed'],loaded.seed)
            with self.assertRaises(FileExistsError):
                exec(cell('mock_maker_interface.ipynb',14),ns)


    def test_public_scenarios_preserve_scatter_and_file_keys(self):
        for label, key, uvx, sarm in [('Current', 'current', 0.24, 0.30),
                                     ('Intermediate', 'future', 0.17, 0.20),
                                     ('Optimistic', 'targeted', 0.14, 0.15)]:
            with self.subTest(scenario=label):
                ns = self.namespace()
                source = cell('mock_maker_interface.ipynb', 8)
                exec(source.replace("scenario = 'Current'", f"scenario = {label!r}"), ns)
                self.assertEqual(ns['errors']['uvx'], key)
                self.assertEqual(ns['errors']['sarm'], key)
                self.assertEqual(resolve_uvx_error(key).sigma_uvx_dex, uvx)
                self.assertEqual(resolve_sarm_error(key).sigma_ln_da, sarm)
                exec(cell('mock_maker_interface.ipynb', 10), ns)
                self.assertEqual(ns['outdir'], Path('mock_data'))
                self.assertTrue(ns['stem'].endswith('_' + key))
                config = cell('sampler_interface.ipynb', 4).split('\nif not mock_file.is_file():')[0]
                sampler_ns = {'Path': Path}
                exec(config.replace("scenario = 'Current'", f"scenario = {label!r}"), sampler_ns)
                self.assertEqual(sampler_ns['mock_file'].name,
                                 f'M2alpha-0.07beta-0.014nQSO1000_{key}_paired_dataset.npz')

    def test_notebook_plotting_has_no_private_style_or_tex_requirement(self):
        import ast
        style = "plt.style.use(Path('tools/qso_gp_mock_matplotlibrc'))"
        self.assertTrue((ROOT / 'tools/qso_gp_mock_matplotlibrc').is_file())
        for name in ['mock_maker_interface.ipynb', 'sampler_interface.ipynb']:
            self.assertIn(style, cell(name, 2))
            self.assertNotIn('sanglier', cell(name, 2))
        for index in [10, 14]:
            calls = [node for node in ast.walk(ast.parse(cell('sampler_interface.ipynb', index)))
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id == 'PlotConfig']
            self.assertEqual(len(calls), 1)
            tex = next(keyword.value for keyword in calls[0].keywords if keyword.arg == 'usetex')
            self.assertIs(tex.value, False)
