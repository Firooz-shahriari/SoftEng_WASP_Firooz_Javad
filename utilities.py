"""
This module is implemented to inform us about the progress of the algorithms,
during the experimets.
"""
def monitor(name,current,total):
    if (current+1) % (total/10) == 0:
        print ( name + ' %d%% completed' % int(100*(current+1)/total) )

def monitor_testing(name,current,total):
    if (current+1) % (total/10) == 0:
        return f'{name} {int(100*(current+1)/total)}% completed'

def function(name):
    pass
