import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from idsmultiplicity.datasets import NSL_COLUMNS, prepare_dataset, save_prepared, load_prepared
from idsmultiplicity.training import plan_models, run_training
from idsmultiplicity.analysis import compute_disagreement, diagnose_shift, aggregate_runs
from idsmultiplicity.explain import compute_shap
from idsmultiplicity.plot import plot_run

BASE=Path(__file__).resolve().parents[1]/'configs'


def config(name):
    return yaml.safe_load((BASE/f'{name}.yaml').read_text())


def test_training_plan_is_63_with_origin_disjoint_streams():
    a=config('unsw_nb15')['training']
    plan1=plan_models(a,1)
    plan2=plan_models(a,2)
    assert len(plan1)==len(plan2)==63
    assert [sum(s['comparison_block']==b for s in plan1) for b in
            ('baseline','seed','bootstrap','architecture_activation','subset')]==[1,20,20,20,2]
    assert set(s['train_seed'] for s in plan1).isdisjoint(s['train_seed'] for s in plan2)


def test_unsw_loader_prevents_label_leakage(unsw_files,tmp_path):
    cfg=config('unsw_nb15')['dataset']
    cfg['root']=str(unsw_files)
    data=prepare_dataset(cfg,tmp_path/'audit',100)
    assert len(data.y_test)==40
    assert len(data.y_val)==20
    assert all(not any(x in n.lower() for x in ['attack_cat','label','id']) for n in data.feature_names)
    assert data.metadata_test.attack_type.iloc[0]=='Normal'
    assert data.metadata_test.source_row.iloc[-1]==39
    save_prepared(data,tmp_path/'audit')
    restored=load_prepared(tmp_path/'audit')
    assert np.array_equal(restored.x_test,data.x_test)
    assert np.array_equal(restored.y_val,data.y_val)


def test_random_test_fixed_across_origin_seeds(tmp_path):
    root=tmp_path/'test';root.mkdir()
    n=200
    pd.DataFrame({'f':np.arange(n),'Label':['Benign','DDoS']*(n//2)}).to_csv(root/'rows.csv',index=False)
    cfg={'dataset':'ciciot2023','root':str(root),'files':['*.csv'],'label_candidates':['Label'],
         'test_split_seed':42,'validation_fraction':.2, 'test_fraction':.2}
    a=prepare_dataset(cfg,tmp_path/'a',100)
    b=prepare_dataset(cfg,tmp_path/'b',200)
    assert a.metadata_test.source_row.equals(b.metadata_test.source_row)
    assert np.array_equal(a.y_test,b.y_test)
    assert not np.array_equal(a.y_train,b.y_train) or not np.array_equal(a.x_train,b.x_train)


def test_temporal_split_has_no_future_data(tmp_path):
    root=tmp_path/'temporal';root.mkdir()
    n=200
    df=pd.DataFrame({'Timestamp':pd.date_range('2023-01-01',periods=n,freq='min'),
                     'Flow Duration':np.arange(n),'Label':['Benign','Attack']*(n//2)})
    df.to_csv(root/'part.csv',index=False)
    cfg={'dataset':'cse_cic_ids2018','root':str(root),'files':['*.csv'],
         'split':'temporal','time_column':'Timestamp','label_candidates':['Label']}
    data=prepare_dataset(cfg,tmp_path/'out',42)
    assert pd.to_datetime(data.metadata_train.Timestamp).max()<pd.to_datetime(data.metadata_val.Timestamp).min()
    assert pd.to_datetime(data.metadata_val.Timestamp).max()<pd.to_datetime(data.metadata_test.Timestamp).min()
    assert not any('Timestamp' in x for x in data.feature_names)


def test_all_csv_profiles_on_small_local_fixtures(tmp_path):
    # These tests validate loader plumbing, not source-schema guarantees.
    for name in ['nf_unsw_nb15_v3','nf_ton_iot_v3','nf_bot_iot_v3','nf_cse_cic_ids2018_v3',
                 'ciciot2023','ciciomt2024','bccc_cse_cic_ids2018']:
        root=tmp_path/name
        root.mkdir()
        cfg=config(name)['dataset']
        cfg['root']=str(root)
        n=80
        df=pd.DataFrame({'Feature':np.arange(n), 'Label':[0,1]*(n//2) if name.startswith('nf_') else
                         ['Benign','DDoS']*(n//2), 'Attack':['Benign','DDoS']*(n//2)})
        if name=='ciciomt2024':
            (root/'wifi'/'train').mkdir(parents=True)
            (root/'wifi'/'test').mkdir(parents=True)
            df.to_csv(root/'wifi'/'train'/'flow.csv',index=False)
            df.to_csv(root/'wifi'/'test'/'flow.csv',index=False)
        else:
            df.to_csv(root/'flow.csv',index=False)
        data=prepare_dataset(cfg,tmp_path/f'out-{name}',42)
        assert data.x_train.shape[1]>=1
        assert data.y_test.sum()>0


def test_full_smoke_training_disagreement_shap(unsw_files,tmp_path):
    torch.set_num_threads(1)
    cfg=config('smoke')
    cfg['dataset']['root']=str(unsw_files)
    cfg['run_dir']=str(tmp_path/'runs'/'seed{origin_seed}')
    run=run_training(cfg,1)
    metrics=pd.read_csv(run/'metrics.csv')
    assert len(metrics)==5
    assert (run/'models').is_dir()
    assert len(list((run/'models').glob('*.pt')))==5
    # Idempotent resume.
    run_training(cfg,1)
    assert len(pd.read_csv(run/'metrics.csv'))==5
    analysis=compute_disagreement(run,tolerance=1.)
    assert len(pd.read_csv(analysis/'rashomon_models.csv'))==5
    samples=pd.read_csv(analysis/'sample_disagreement.csv')
    assert len(samples)==40
    assert set(samples.consensus_state.unique())<= {'contested','unanimous_normal','unanimous_attack'}
    assert (diagnose_shift(run)/'feature_shift.csv').exists()
    s=compute_shap(run,tolerance=1.,max_background=4,max_explain=6,device='cpu',top_k=3)
    shap_summary=json.loads((s/'explanation_summary.json').read_text())
    assert shap_summary['n_models']==5
    assert shap_summary['n_explained_samples']==6
    assert (s/'global_explanation_pairs.csv').exists()
    figures=plot_run(run,tolerance=1.)
    assert (figures/"validation_test_gap.pdf").exists()
    out=aggregate_runs([run],tolerance=1.,out_dir=tmp_path/'reports')
    assert (out/'origin_results.csv').exists()


def test_nsl_headerless_files_and_attack_metadata(tmp_path):
    root=tmp_path/'nsl_kdd';root.mkdir()
    def write(path,n):
        records=[]
        for i in range(n):
            record={c:0 for c in NSL_COLUMNS}
            record.update({'protocol_type':'tcp' if i%2 else 'udp',
                           'service':'http','flag':'SF','label':'normal' if i%2 else 'neptune',
                           'difficulty':4, 'duration':i})
            records.append(record)
        pd.DataFrame(records)[NSL_COLUMNS].to_csv(path,index=False,header=False)
    write(root/'KDDTrain+.txt',100)
    write(root/'KDDTest+.txt',40)
    cfg=config('nsl_kdd')['dataset'];cfg['root']=str(root)
    data=prepare_dataset(cfg,tmp_path/'out',42)
    assert len(data.y_test)==40
    assert set(data.metadata_test.attack_type)=={'normal','neptune'}
    assert not any('difficulty' in c.lower() for c in data.feature_names)


def test_mismatched_csv_feature_schemas_raise(tmp_path):
    root=tmp_path/'mixed';root.mkdir()
    pd.DataFrame({'Label':['Benign','Attack']*20,'x':range(40)}).to_csv(root/'a.csv',index=False)
    pd.DataFrame({'Label':['Benign','Attack']*20,'z':range(40)}).to_csv(root/'b.csv',index=False)
    cfg={'dataset':'ciciot2023','root':str(root),'files':['*.csv'],'label_candidates':['Label']}
    with pytest.raises(ValueError,match='different schemas'):
        prepare_dataset(cfg,tmp_path/'out',42)
