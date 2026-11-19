import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def unsw_files(tmp_path):
    root=tmp_path/'unsw_nb15';root.mkdir()
    rng=np.random.default_rng(123)
    def make(n):
        label=np.array([0,1]*(n//2))
        return pd.DataFrame({'id':np.arange(n),'proto':np.where(label==0,'tcp','udp'),
                             'service':np.where(label==0,'http','dns'),
                             'state':np.where(label==0,'FIN','CON'),
                             'dur':rng.normal(label,1,n),
                             'sbytes':rng.normal(label*.5,1,n),
                             'dbytes':rng.normal(0,1,n),
                             'attack_cat':np.where(label==0,'Normal','Reconnaissance'),
                             'label':label})
    make(100).to_csv(root/'UNSW_NB15_training-set.csv',index=False)
    make(40).to_csv(root/'UNSW_NB15_testing-set.csv',index=False)
    return root
