"""Check a chosen official data file's columns/labels before a full experiment."""
import argparse
from collections import Counter
from pathlib import Path
import yaml
from idsmultiplicity.datasets import _load_raw

p=argparse.ArgumentParser()
p.add_argument('--config',required=True)
args=p.parse_args()
cfg=yaml.safe_load(Path(args.config).read_text())['dataset']
train,test,details=_load_raw(cfg,int(cfg.get('sampling_seed',42)))
print('Dataset:',cfg['dataset'])
print('Protocol:',details['method'])
for name,df in [('training pool',train),('official test',test)]:
 if df is None: continue
 print('\n'+name,df.shape)
 print('Counts:',df.__target.value_counts().to_dict())
 print('Attack labels:',Counter(df.__attack_type).most_common(20))
 print('Columns:',list(df.columns))
 print('Missing:',df.isna().sum().sort_values(ascending=False).head(10).to_dict())
