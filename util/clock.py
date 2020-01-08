import time

def tic():
    globals()['tt'] = time.clock()

def toc(str='take',silence=False):
    t = time.clock()-globals()['tt']
    if silence == False:
        print('\n%s time: %.8f seconds\n' % (str, t))
    return t