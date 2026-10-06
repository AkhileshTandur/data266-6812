"""Build a standalone notebook from the percent-format HW6 source."""
import json
from pathlib import Path

source = Path('hw6.py').read_text(encoding='utf-8')
cells = []
for chunk in source.split('# %%')[1:]:
    header, _, body = chunk.partition('\n')
    if '[markdown]' in header:
        body = '\n'.join(line[2:] if line.startswith('# ') else
                         '' if line == '#' else line for line in body.splitlines()).strip()
        cell = dict(cell_type='markdown', metadata={}, source=body.splitlines(keepends=True))
    else:
        cell = dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                    source=body.strip().splitlines(keepends=True))
    cell['id'] = f'hw6-{len(cells):02d}'
    cells.append(cell)
notebook = dict(cells=cells, metadata=dict(
    kernelspec=dict(display_name='Python 3', language='python', name='python3'),
    language_info=dict(name='python', version='3.13.0'),
    accelerator='GPU'), nbformat=4, nbformat_minor=5)
Path('HW6_STL10_Representation_Learning.ipynb').write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Built standalone HW6 notebook: {len(cells)} cells')
