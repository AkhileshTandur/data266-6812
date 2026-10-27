"""Execute all notebook cells with the current Python interpreter."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

root = Path(__file__).resolve().parent
path = root / 'HW8.ipynb'
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name='python3')
manager._kernel_spec = KernelSpec(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                                  display_name='Python 3', language='python')
client = NotebookClient(notebook, timeout=14400, km=manager,
                        resources={'metadata': {'path': str(root)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
nbformat.validate(notebook)
print('HW8 notebook executed and saved.')
