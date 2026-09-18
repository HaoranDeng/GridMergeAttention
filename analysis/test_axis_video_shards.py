"""Verify disjoint coverage, duplicate rejection, and explicit speed exception."""
import json
from pathlib import Path
import tempfile
import unittest
from analysis.axis_video_shards import publish
from analysis.axis_c4_speed_gate import verify_gate
from analysis.quick_axis_c4_qkv import save,sha
from utils.prompt_contract import build_prompt_manifest,ensure_output_prompt_manifest


class ShardTests(unittest.TestCase):
    def test_union_and_duplicate(self):
        root=Path(__file__).resolve().parents[1]
        prompt_path=root/'prompts/MovieGenVideoBench_extended.txt'
        prompts=[s.strip() for s in prompt_path.read_text().splitlines() if s.strip()]
        manifest=build_prompt_manifest(prompt_path,prompts,selected_count=20,seed=0,num_samples=1,vbench_filenames=False,allow_noncanonical=False)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);sources=[base/'a',base/'b']
            protocol=dict(source_sha256={'test':'digest'},steps=50,seed=0,axes='tw',capacity=4,height=720,width=1280,frames=81,expected_calls_per_video=4000)
            def write(source,indices):
                ensure_output_prompt_manifest(source,manifest)
                rows=[]
                for i in indices:
                    name=f'{i:03d}-00.mp4';(source/name).write_bytes(b'test placeholder; never decoded')
                    rows.append(dict(index=i,video=name,seed=0,attention_calls=4000,layer_calls=[100]*40,dense_self_attention_calls=0))
                save(source/'runtime.json',dict(protocol=protocol,videos=rows))
            write(sources[0],[0,2]);write(sources[1],[1,3])
            config=base/'config.json';save(config,dict(axes='tw',output=str(base/'union'),sources=list(map(str,sources))))
            result=publish(config)
            self.assertEqual([v['index'] for v in result['videos']],[0,1,2,3])
            self.assertFalse(result['generation_complete'])
            self.assertFalse(result['first5_quality_complete'])
            write(sources[1],[1,2,3])
            with self.assertRaisesRegex(AssertionError,'Duplicate'):publish(config)

    def test_speed_exception_is_explicit_and_th_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'kernel').write_text('frozen kernel')
            method=dict(validated=True,exclusive_gpu=True,module='attention.axis_current_c4_cuda',
                        samples=[dict(input_kind='captured_real_qkv',method_median_ms=549,dense_median_ms=328)])
            gate=dict(passed=False,methods={'th':method},scope='full attention including layouts and all40 heads',
                      dtype='bfloat16',shape=[1,40,75600,128],source_sha256={'kernel':sha(root/'kernel')},
                      user_speed_exception=dict(axes='th',instruction='Run the slower T/H comparison'))
            path=root/'gate.json';save(path,gate)
            with self.assertRaises(AssertionError):verify_gate(path,root,'th')
            verify_gate(path,root,'th',True)
            with self.assertRaises(AssertionError):verify_gate(path,root,'tw',True)
            (root/'kernel').write_text('modified kernel')
            with self.assertRaisesRegex(AssertionError,'stale'):verify_gate(path,root,'th',True)


if __name__=='__main__':unittest.main()
