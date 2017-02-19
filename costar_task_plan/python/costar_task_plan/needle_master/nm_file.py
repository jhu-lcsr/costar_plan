
'''
file utilities
'''


def ParseEnvironmentName(filename):
    toks = filename.split('/')[-1].split('.')[0].split('_')
    return toks[1]


def ParseDemoName(filename):
    toks = filename.split('/')[-1].split('.')[0].split('_')
    print toks
    return (int(toks[1]), toks[2])
