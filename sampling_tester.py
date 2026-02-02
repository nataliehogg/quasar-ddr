from cobaya.run import run
from likelihood import DDRlike

Afid = 1
xi0fid = 0

info = {'params': {
                  'A': {'prior': {'min': -1., 'max': 1.},
                            'ref': {'dist': 'norm', 'loc': Afid, 'scale': 0.1},
                            'proposal': 0.05,
                            'latex': 'A'},
                   'xi0': {'prior': {'min': -1., 'max': 1.},
                            'ref': {'dist': 'norm', 'loc': xi0fid, 'scale': 0.1},
                            'proposal': 0.05,
                            'latex': 'xi'},
                   'H0': {'prior': {'min': 50., 'max': 90.},
                            'ref': {'dist': 'norm', 'loc': 67, 'scale': 0.1},
                            'proposal': 0.1,
                            'latex': 'H0'}
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
