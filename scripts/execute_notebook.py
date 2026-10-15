from pathlib import Path
import os,json,time
import nbformat
from nbclient import NotebookClient
r=Path(__file__).resolve().parents[1]
nb=nbformat.read(r/'notebooks/policylab.ipynb',as_version=4)
start=time.perf_counter()
nb=NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(r)}}).execute()
errors=[o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
assert not errors
out=r.parents[1]/'publish-kit/policylab-release/policylab-executed.ipynb';nbformat.write(nb,out)
receipt={'code_cells':sum(c.cell_type=='code' for c in nb.cells),'errors':len(errors),'seconds':time.perf_counter()-start,'scope':'Frozen real-audit, tests, hashes and synthetic rerun; full raw-source optional switch off'}
(r/'artifacts/notebook-validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2))
