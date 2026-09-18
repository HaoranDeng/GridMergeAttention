"""CPU-only summary of generated H40 SASS scheduling and local accesses."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument("compile_json",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    report=json.loads(a.compile_json.read_text())
    results={}
    for name,c in report["candidates"].items():
        if not c["compiled"]:
            continue
        text=subprocess.check_output(["/usr/local/cuda/bin/cuobjdump","--dump-sass",c["module"]],text=True)
        usage=subprocess.check_output(["/usr/local/cuda/bin/cuobjdump","--dump-resource-usage",c["module"]],text=True)
        resource_line=re.search(r'Function _Z12merge_kernelILi21ELi40[^\n]*\n\s*([^\n]+)',usage)
        assert resource_line is not None
        static_resources={key:int(value) for key,value in
                          re.findall(r'([A-Z]+(?:\[\d+\])?):(\d+)',resource_line.group(1))}
        part=next(x for x in text.split("Function : ")[1:] if x.splitlines()[0].startswith("_Z12merge_kernelILi21ELi40E"))
        ins=[]
        for line in part.splitlines():
            m=re.search(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*;",line)
            if m:
                ins.append({"address":m[1],"instruction":m[2]})
        counts={key:sum(key in x["instruction"] for x in ins) for key in
                ("HGMMA","WARPGROUP.DEPBAR","WARPGROUP.ARRIVE","MUFU","LDL","STL","UTMASTG","R2UR","BRA.U","SETMAXNREG")}
        # Bound output to the intervals ending at the first six WGMMA waits.
        intervals=[]
        last=0
        for i,x in enumerate(ins):
            if "WARPGROUP.DEPBAR" in x["instruction"]:
                segment=ins[last:i]
                matrices=[j for j,v in enumerate(segment) if "HGMMA" in v["instruction"]]
                after=segment[matrices[-1]+1:] if matrices else []
                in_flight=segment[matrices[0]+1:] if matrices else []
                intervals.append({"wait":x,"instructions_since_previous_wait":len(segment),
                                  "matrix_instructions":len(matrices),
                                  "loads_between_first_matrix_and_wait":[v for v in in_flight if "LDG" in v["instruction"]],
                                  "mufu_after_last_matrix_before_wait":sum("MUFU" in v["instruction"] for v in after),
                                  "epilogue_between_matrix_and_wait":{key:sum(
                                      bool(re.search(r"\bSTG(?:\.|\s)",v["instruction"])) if key=="STG"
                                      else key in v["instruction"] for v in after)
                                      for key in ("STSM","UTMASTG","LDG","F2FP","STG","FADD","FMUL","FFMA")},
                                  "tail":segment[-5:]})
                last=i+1
                if len(intervals)==6:
                    break
        # A loop-carried QK issue can wait at the loop head on the next
        # iteration. Record the static tail separately; it is not another
        # acyclic matrix-to-wait interval.
        final_wait=max(i for i,x in enumerate(ins) if 'WARPGROUP.DEPBAR' in x['instruction'])
        tail=ins[final_wait+1:]
        matrices=[i for i,x in enumerate(tail) if 'HGMMA' in x['instruction']]
        after=tail[matrices[-1]+1:] if matrices else []
        results[name]={"opcode_counts":counts,"first_wait_intervals":intervals,
                       "static_resources":static_resources,
                       "register_adjustments":[x for x in ins if 'SETMAXNREG' in x['instruction']],
                       "instruction_sha256":hashlib.sha256('\n'.join(x['instruction'] for x in ins).encode()).hexdigest(),
                       "static_tail_after_final_wait":{
                           "matrices":len(matrices),
                           "after_last_matrix":{key:sum(key in x['instruction'] for x in after)
                                                for key in ('STSM','UTMASTG','LDG','F2FP','BRA')}}}
    a.output.write_text(json.dumps(results,indent=2))
    print(json.dumps({k:v["opcode_counts"] for k,v in results.items()}))


if __name__=="__main__":
    main()
