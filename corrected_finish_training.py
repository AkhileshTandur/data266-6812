"""Finish controlled-batch comparisons and the resident boundary after the suite."""
import json
from pathlib import Path
import torch
import corrected_suite as suite

ROOT=Path(__file__).resolve().parent
if 'CORRECTED TRAINING SUITE COMPLETE' not in (ROOT/'corrected_suite_linux.log').read_text():
    raise RuntimeError('Wait for the training suite to finish; never compete for GPU timing')
suite.FOLDER=ROOT/'corrected'/torch.cuda.get_device_name().replace(' ','_')
suite.run('baseline',512)
for technique in ['amp','checkpoint','adam8bit','amp8bit','combined']:
    suite.run(technique,32)
suite.log('CORRECTED CONTROLLED-BATCH FOLLOWUP COMPLETE')
