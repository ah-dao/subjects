"""Exact weighted Fisher-Jenks 1D SSE partition using monotone divide-and-conquer DP."""
import numpy as np


def natural_breaks(score, classes=5):
    score = np.asarray(score,dtype='float64')
    if score.ndim!=1 or not len(score) or not np.isfinite(score).all():
        raise ValueError('Expected finite one-dimensional scores')
    if not isinstance(classes,int) or classes<2:
        raise ValueError('At least two classes required')
    values, counts = np.unique(score,return_counts=True)
    n = len(values)
    if n<classes: raise ValueError('Not enough distinct scores for requested classes')
    # Repeated values are indivisible and weighted by their unit counts. No subsampling.
    z = (values-values[0])/(values[-1]-values[0])
    weights = np.r_[0,np.cumsum(counts,dtype='float64')]
    sums = np.r_[0,np.cumsum(counts*z,dtype='float64')]
    squares = np.r_[0,np.cumsum(counts*z*z,dtype='float64')]
    previous = np.full(n+1,np.inf); previous[0] = 0.
    parents = np.full((classes+1,n+1),-1,dtype='int64')
    for k in range(1,classes+1):
        current = np.full(n+1,np.inf)
        def solve(left,right,opt_left,opt_right):
            if left>right: return
            mid = (left+right)//2
            starts = np.arange(opt_left,min(mid-1,opt_right)+1)
            mass = weights[mid]-weights[starts]
            total = sums[mid]-sums[starts]
            cost = np.maximum(0.,squares[mid]-squares[starts]-total*total/mass)
            choices = previous[starts]+cost
            chosen = int(starts[np.argmin(choices)])
            current[mid] = choices[chosen-opt_left]
            parents[k,mid] = chosen
            solve(left,mid-1,opt_left,chosen)
            solve(mid+1,right,chosen,opt_right)
        solve(k,n,k-1,n-1)
        previous = current
    boundaries = []
    end = n
    for k in range(classes,1,-1):
        start = int(parents[k,end])
        boundaries.append(values[start-1])
        end = start
    boundaries = np.asarray(boundaries[::-1])
    # Jenks thresholds are inclusive class maxima: a score equal to a break stays below it.
    labels = np.searchsorted(boundaries,score,side='left')+1
    if not np.array_equal(np.unique(labels),np.arange(1,classes+1)):
        raise ValueError('Collapsed natural-break classes')
    return labels,boundaries
