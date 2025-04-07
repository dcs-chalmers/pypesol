
### Predictors: forecast input based on historical data ###

"""
predictor(table,t,i0,imax)
            function for predicting t-th entry in the table,
            i0 being the prediction being made and imax maximum for t
"""

def truth_predictor(table, t, i0=None, imax=None):
    
    return table[t]

def average_predictor(table, t, i0=None, imax=None, nbdays=7):
    return sum(table[(t-24*i)%len(table)] for i in range(1,nbdays+1)) / nbdays

def previous_predictor(table, t, i0, imax, nbdays=7):
    return table[(t-1)%len(table)]

def mix_predictor(table, t, i0, imax, nbdays=7):
    base = table[(t-1)%len(table)] + (table[(t-1)%len(table)]-table[(t-2)%len(table)])
    if imax == i0:
        return base
    return base*(imax-t)/(imax-i0) + average_predictor(table, t) * (t-i0)/(imax-i0)

def mix2_predictor(table, t, i0, imax, nbdays=7):
    base = table[(t-1)%len(table)]
    if imax == i0:
        return base
    return base*(imax-t)/(imax-i0) + average_predictor(table, t) * (t-i0)/(imax-i0)

def linear_predictor(table, t, i0, imax, basis=average_predictor):
    if imax == i0:
        return table[t]
    else:
        return table[t]*(imax-t)/(imax-i0) + basis(table, t) * (t-i0)/(imax-i0)
