"""CPU-only proof of bounded capture completion and parent environment."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analysis import launch_full40_bf16_capture as launcher


class Full40CaptureGuardTests(unittest.TestCase):
    def make_complete(self,root):
        target=root/'captures';target.mkdir()
        captures=[]
        for p in range(3):
            for c in (0,33,40,73,80,113,120,153):
                name=f'prompt{p:03d}_call{c:03d}_layer{c%40}.pt'
                path=target/name;path.write_bytes(name.encode())
                captures.append({'capture_file':name,'bytes':path.stat().st_size,'sha256':launcher.sha(path),
                    'shape':[1,75600,40,128],'heads':list(range(40)),'dtype':'torch.bfloat16'})
        report={'complete':True,'total_self_attention_calls':480,'total_captures':24,
            'checkpoint_weights':'raw_state_dict','actual_forced_sdpa_backend':'flash',
            'actual_use_torch_rmsnorm':False,'captures':captures,
            'records':[{'prompt_index':p,'call_index':c,'layer_index':c%40,'step_index':c//40}
                       for p in range(3) for c in range(160)]}
        (target/'exit_code').write_text('0\n')
        launcher.write(target/'capture_manifest.json',report)
        return target,report

    def test_all_steps_all_heads_and_hashes_are_required(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);target,report=self.make_complete(root)
            self.assertEqual(len(launcher.verify_complete(root)['captures']),24)
            for key,bad in (('total_captures',12),('total_self_attention_calls',160),
                            ('checkpoint_weights','ema'),('actual_use_torch_rmsnorm',True)):
                launcher.write(target/'capture_manifest.json',report | {key:bad})
                with self.assertRaises(ValueError):launcher.verify_complete(root)
            launcher.write(target/'capture_manifest.json',report)
            (target/report['captures'][0]['capture_file']).write_bytes(b'changed')
            with self.assertRaises(ValueError):launcher.verify_complete(root)

    def test_no_duplicate_metadata_or_reordered_dense_calls(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);target,report=self.make_complete(root)
            changed=json.loads(json.dumps(report))
            changed['captures'][1]=changed['captures'][0]
            launcher.write(target/'capture_manifest.json',changed)
            with self.assertRaisesRegex(ValueError,'metadata coverage'):launcher.verify_complete(root)
            changed=json.loads(json.dumps(report));changed['records'][0],changed['records'][1]=changed['records'][1],changed['records'][0]
            launcher.write(target/'capture_manifest.json',changed)
            with self.assertRaisesRegex(ValueError,'call order'):launcher.verify_complete(root)
            changed=json.loads(json.dumps(report));changed['captures'][0]['dtype']='torch.float16'
            launcher.write(target/'capture_manifest.json',changed)
            with self.assertRaisesRegex(ValueError,'BF16'):launcher.verify_complete(root)

    def test_capture_environment_preserves_forced_flash_and_native_rmsnorm(self):
        env={'TOKEN_MERGING_FORCE_TORCH_RMSNORM':'1','WAN_SDPA_BACKEND':'flash','CUDA_VISIBLE_DEVICES':'GPU-test'}
        with patch.object(launcher.guard,'child_environment',return_value=env.copy()):
            actual=launcher.child_environment('unused','GPU-test')
        self.assertNotIn('TOKEN_MERGING_FORCE_TORCH_RMSNORM',actual)
        self.assertEqual(actual['WAN_SDPA_BACKEND'],'flash')
        self.assertEqual(actual['FULL40_CAPTURE_PARENT'],str(os.getpid()))

    def test_no_overwrite_or_queue(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            args=SimpleNamespace(output=root,source_root=root,gpu=5)
            with patch.object(launcher.guard,'require_clean') as clean:
                with self.assertRaisesRegex(ValueError,'fresh output'):launcher.launch(args)
                clean.assert_not_called()
            args.output=root/'new'
            with patch.object(launcher.guard,'require_clean',side_effect=ValueError('busy')) as clean:
                with self.assertRaisesRegex(ValueError,'busy'):launcher.launch(args)
                clean.assert_called_once_with(5)
                self.assertFalse(args.output.exists())


if __name__=='__main__':unittest.main()
