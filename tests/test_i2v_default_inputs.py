import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('i2v_inputs', ROOT / 'analysis/build_i2v_prompt_set.py')
inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inputs)


class I2VDefaultInputs(unittest.TestCase):
    def test_default_matches_the_frozen_experiment(self):
        default = inputs.build()
        path = ROOT / 'experiments/q80k90_i2v_20260924/campaigns/replace019_allpoints/experiment_plan.json'
        frozen = json.loads(path.read_text())['input_manifest']
        self.assertEqual(len(default['items']), 20)
        for row, old in zip(default['items'], frozen):
            self.assertEqual(row['prompt_en'], old['prompt'])
            self.assertEqual(row['image_sha256'], old['image_sha256'])
            self.assertEqual(row['prompt_sha256'], old['prompt_sha256'])
            self.assertEqual(row['seed'], old['seed'])

    def test_only019_changes_and_history_is_explicit(self):
        default = inputs.build()
        legacy = inputs.build('legacy-stratified')
        for a, b in zip(default['items'][:19], legacy['items'][:19]):
            for key in ('index', 'prompt_en', 'image_name', 'image_type'):
                self.assertEqual(a[key], b[key])
        self.assertEqual(default['items'][19]['prompt_en'], 'a person holding a sparkler in their hand')
        self.assertEqual(legacy['items'][19]['prompt_en'], 'a shadow of a hand reaching for a leaf')
        self.assertEqual(default['items'][19]['source_index'], 956)
        self.assertTrue(default['selection_history']['post_hoc_replacement'])

    def test_reject_changed_prompt_and_image(self):
        changed = inputs.build()
        changed['items'][19]['prompt_en'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Prompt hash'):
            inputs.validate(changed)
        with tempfile.TemporaryDirectory() as folder:
            manifest = {'items': [dict(index=0, seed=0, prompt_en='test', image_name='test.jpg',
                                       image_sha256=hashlib.sha256(b'expected').hexdigest())]}
            Path(folder, 'test.jpg').write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'Image hash'):
                inputs.validate(manifest, Path(folder))

    def test_output_pairing_and_seed_order(self):
        manifest = inputs.build()
        with tempfile.TemporaryDirectory() as folder:
            stem = Path(folder) / 'default'
            inputs.write(manifest, stem)
            self.assertEqual(Path(str(stem) + '.txt').read_text().splitlines(),
                             [r['prompt_en'] for r in manifest['items']])
            self.assertEqual(Path(str(stem) + '.images.txt').read_text().splitlines(),
                             [r['image_name'] for r in manifest['items']])
            self.assertEqual([r['seed'] for r in manifest['items']], list(range(20)))

    def test_larger_set_requires_explicit_policy(self):
        with self.assertRaisesRegex(ValueError, 'fixed default'):
            inputs.build(n=50)


if __name__ == '__main__':
    unittest.main()
