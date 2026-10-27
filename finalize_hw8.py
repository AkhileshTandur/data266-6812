"""Sync the finished discussion into the executed notebook without clearing outputs."""
from pathlib import Path
import csv
import json
import nbformat

report = Path('HW8.md').read_text(encoding='utf-8')
nb = nbformat.read('HW8.ipynb', as_version=4)
assert nb.cells[-1].cell_type == 'markdown'
nb.cells[-1].source = report[report.index('## Discussion'):]
nbformat.validate(nb)
code = [cell for cell in nb.cells if cell.cell_type == 'code']
assert all(cell.execution_count is not None for cell in code)
assert not any(output.output_type == 'error' for cell in code for output in cell.outputs)
images = sum('image/png' in output.get('data', {}) for cell in code for output in cell.outputs)
assert images == 3, images
assert 'Training is in progress' not in report
nbformat.write(nb, 'HW8.ipynb')
metrics = json.loads(Path('hw8_outputs/metrics.json').read_text())
assert metrics['epochs'] == 50 and len(metrics['loss']) == 50
with Path('hw8_outputs/training_history.csv').open('w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['epoch', 'noise_prediction_mse'])
    writer.writerows(enumerate(metrics['loss'], 1))
print(f'PASS: {len(code)} executed code cells, {images} embedded figures, 50 training epochs, no notebook errors, final discussion saved.')
