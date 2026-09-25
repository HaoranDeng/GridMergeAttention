"""Derive disjoint-index worker without changing any frozen attention source."""
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    s=(root/'analysis/run_axis_c4_videos.py').read_text()
    s=s.replace('    a = p.parse_args()', '''    p.add_argument('--indices',nargs='+',type=int,required=True)
    p.add_argument('--aggregate-config',type=Path,required=True)
    a = p.parse_args()
    assert a.indices and len(set(a.indices))==len(a.indices)
    assert all(0<=i<20 for i in a.indices)
    from analysis.axis_video_shards import publish,evaluate_ready
    assert not (a.output/'runtime.json').exists(),'Fresh shard output required'
''')
    s=s.replace("'first_quality_at':5}", "'first_quality_at':5,'shard_indices':a.indices}")
    s=s.replace('for index in range(20):','for index in a.indices:')
    s=s.replace("            if index==4:\n                evaluate_count(5)\n                runtime['first5_quality_complete']=True;save(runtime_path,runtime)",
                "            evaluate_ready(a.aggregate_config,a.reference,prompt_path,prompts,a.axes)")
    s=s.replace("    runtime['generation_complete'] = True; save(runtime_path,runtime)",
                "    runtime['shard_generation_complete'] = True; save(runtime_path,runtime)")
    s=s.replace("    evaluate_count(20)\n    runtime['quality_complete'] = True; save(runtime_path,runtime)",
                "    evaluate_ready(a.aggregate_config,a.reference,prompt_path,prompts,a.axes)")
    start=s.index('    def evaluate_count(count):');end=s.index('    with torch.inference_mode():',start)
    s=s[:start]+s[end:]
    (root/'analysis/run_axis_c4_video_shard.py').write_text(s)

if __name__=='__main__':main()
