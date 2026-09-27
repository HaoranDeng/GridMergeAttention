"""CPU checks for ownership handoff without restarting or duplicating GPU work."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import worker


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.old_gpu={'uuid':'dense-gpu'};self.new_gpu={'uuid':'c4-gpu'}
        worker.save(self.root/'experiment_plan.json',{'gpu':self.old_gpu,'snapshot':str(self.root)})
        worker.save(self.root/'parallel_plan.json',dict(original_plan_sha256=worker.sha(self.root/'experiment_plan.json'),
                    methods=['gate','C4'],max_gpus=2,gpu=self.new_gpu))

    def tearDown(self):
        self.tmp.cleanup()

    def run_mode(self,mode,*extra,**kwargs):
        with patch.object(worker,'ROOT',self.root),patch.object(sys,'argv',['worker.py',mode,*extra]),\
             patch.dict(os.environ,CUDA_VISIBLE_DEVICES='dense-gpu'),\
             patch.object(worker,'verify') as verify,patch.object(worker,'generate') as generate,\
             patch.object(worker.os,'chdir'),contextlib.redirect_stdout(io.StringIO()):
            if 'sleep' in kwargs:
                with patch.object(worker.time,'sleep',kwargs['sleep']):worker.main()
            else:worker.main()
            return verify,generate

    def test_finished_gate_does_not_load_model(self):
        worker.save(self.root/'native_gate.json',dict(complete=True,passed=True,gpu=self.new_gpu))
        verify,generate=self.run_mode('gate')
        verify.assert_not_called();generate.assert_not_called()

    def test_partial_gate_is_not_mistaken_for_pass(self):
        worker.save(self.root/'native_gate.json',dict(complete=False,passed=False,gpu=self.new_gpu))
        sleeps=[]
        def finish(_):
            sleeps.append(True)
            worker.save(self.root/'native_gate.json',dict(complete=True,passed=True,gpu=self.new_gpu))
        _,generate=self.run_mode('gate',sleep=finish)
        self.assertEqual(len(sleeps),1);generate.assert_not_called()

    def test_completed_c4_is_not_generated_twice(self):
        worker.save(self.root/'C4/generation_complete.json',dict(complete=True,videos=50))
        worker.save(self.root/'C4/videos/purity_records.json',[{'gpu':self.new_gpu} for _ in range(50)])
        _,generate=self.run_mode('C4');generate.assert_not_called()

    def test_changed_frozen_plan_is_rejected(self):
        worker.save(self.root/'experiment_plan.json',{'gpu':self.old_gpu,'changed':True})
        with self.assertRaises(AssertionError):self.run_mode('gate')

    def test_parallel_failure_does_not_fall_back_to_duplicate_job(self):
        worker.save(self.root/'parallel/failure.json',{'error':'gate failed'})
        with self.assertRaisesRegex(RuntimeError,'Parallel C4 failed'):self.run_mode('gate')

    def test_new_owner_uses_new_gpu_and_dense_keeps_original(self):
        _,generate=self.run_mode('C4','--parallel')
        self.assertEqual(generate.call_args.args[0]['gpu'],self.new_gpu)
        _,generate=self.run_mode('Dense')
        self.assertEqual(generate.call_args.args[0]['gpu'],self.old_gpu)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.plan=dict(protocol={'steps':50},input_manifest=[{'index':i,'seed':i} for i in range(50)],gpu={'uuid':'c4-gpu'})
        worker.save(self.root/'prompt_manifest.json',dict(protocol=self.plan['protocol'],inputs=self.plan['input_manifest']))
        records=[];proof=[]
        for i in range(2):
            name=f'{i:03d}-00.mp4';(self.root/name).write_bytes(f'fixture-{i}'.encode())
            records.append(dict(output=name,prompt_index=i,sample_index=0,attention_calls=4000,full_attention_calls=4000))
            proof.append(dict(video=name,video_sha256=worker.sha(self.root/name),gpu=self.plan['gpu'],
                              attention_calls=4000,layer_calls=[100]*40,dense_self_attention_calls=0))
        worker.save(self.root/'runtime.json',{'records':records});worker.save(self.root/'purity_records.json',proof)

    def tearDown(self):
        self.tmp.cleanup()

    def test_verified_prefix_is_preserved(self):
        before={p.name:p.read_bytes() for p in self.root.iterdir()}
        self.assertEqual(len(worker.validate_resume(self.plan,self.root)),2)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir()})

    def test_changed_output_is_rejected(self):
        (self.root/'001-00.mp4').write_bytes(b'changed')
        with self.assertRaises(AssertionError):worker.validate_resume(self.plan,self.root)

    def test_changed_seed_is_rejected(self):
        manifest=worker.read(self.root/'prompt_manifest.json');manifest['inputs'][0]['seed']=99
        worker.save(self.root/'prompt_manifest.json',manifest)
        with self.assertRaises(AssertionError):worker.validate_resume(self.plan,self.root)

    def test_unrecorded_video_is_rejected(self):
        (self.root/'002-00.mp4').write_bytes(b'no audited record')
        with self.assertRaises(AssertionError):worker.validate_resume(self.plan,self.root)

    def test_incomplete_attention_calls_are_rejected(self):
        data=worker.read(self.root/'runtime.json');data['records'][1]['attention_calls']=3999
        worker.save(self.root/'runtime.json',data)
        with self.assertRaises(AssertionError):worker.validate_resume(self.plan,self.root)


if __name__=='__main__':
    unittest.main()
