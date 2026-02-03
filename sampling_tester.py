from cobaya.run import run
from likelihood import DDRlike

Afid = 1
Omegam_fid = 0.314

info = {'params': {
                  'A': {'prior': {'min': -1., 'max': 1.},
                            'ref': {'dist': 'norm', 'loc': Afid, 'scale': 0.1},
                            'proposal': 0.05,
                            'latex': 'A'},
                   'H0': {'prior': {'min': 50., 'max': 90.},
                            'ref': {'dist': 'norm', 'loc': 67, 'scale': 0.1},
                            'proposal': 0.1,
                            'latex': 'H0'},
                   'Omegam': {'prior': {'min': 0.1, 'max': 0.5},
                            'ref': {'dist': 'norm', 'loc': Omegam_fid, 'scale': 0.05},
                            'proposal': 0.01,
                            'latex': r'\Omega_{\rm m}'}
                            },
        # 'theory': {'camb': None},
        'sampler': {'mcmc': {'max_tries': 100}},
        'output': 'chains/test',
        'likelihood': {'DDRlike': DDRlike},
        'force': True
        }


info['debug'] = True

#Run everything
updated_info, sampler = run(info)
