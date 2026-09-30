from __future__ import annotations

import numpy as np


class WeightedStumpBoosting:
    """Weighted squared-error stumps; prefix sums avoid repeated window scans."""
    def __init__(self,n_estimators=260,learning_rate=.035,max_thresholds=32,feature_fraction=.85,seed=20260929):
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.max_thresholds=max_thresholds
        self.feature_fraction=feature_fraction
        self.seed=seed

    def fit(self,x,y,sample_weight=None):
        w=np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight,dtype=float)
        w=w/w.sum()
        self.initial=float(np.dot(w,y))
        prediction=np.full(len(y),self.initial)
        rng=np.random.default_rng(self.seed)
        self.stumps=[]
        orders=[np.argsort(x[:,j],kind="stable") for j in range(x.shape[1])]
        cuts=[np.unique(np.quantile(x[:,j],np.linspace(.05,.95,self.max_thresholds))) for j in range(x.shape[1])]
        positions=[np.searchsorted(x[orders[j],j],cuts[j],side="right") for j in range(x.shape[1])]
        cumulative_weight=[np.cumsum(w[o]) for o in orders]
        for _ in range(self.n_estimators):
            residual=y-prediction
            total=float(np.dot(w,residual))
            best=None
            cols=np.sort(rng.choice(x.shape[1],max(1,int(np.ceil(x.shape[1]*self.feature_fraction))),replace=False))
            for j in cols:
                pos=positions[j]
                valid=(pos>0)&(pos<len(y))
                if not valid.any():
                    continue
                n=pos[valid]-1
                sums=np.cumsum(w[orders[j]]*residual[orders[j]])[n]
                wl=cumulative_weight[j][n]
                wr=1-wl
                gain=sums*sums/wl+(total-sums)**2/wr
                k=int(np.argmax(gain))
                candidate=(float(gain[k]),int(j),float(cuts[j][valid][k]),float(sums[k]/wl[k]),float((total-sums[k])/wr[k]))
                if best is None or candidate[0]>best[0]:
                    best=candidate
            if best is None:
                break
            _,j,cut,left,right=best
            prediction+=self.learning_rate*np.where(x[:,j]<=cut,left,right)
            self.stumps.append((j,cut,left,right))
        return self

    def predict(self,x):
        p=np.full(len(x),self.initial)
        for j,cut,left,right in self.stumps:
            p+=self.learning_rate*np.where(x[:,j]<=cut,left,right)
        return p


class SubspaceRidge:
    def __init__(self,seed=20260929,n_estimators=80,alpha=.01,feature_fraction=.65):
        self.seed=seed
        self.n_estimators=n_estimators
        self.alpha=alpha
        self.feature_fraction=feature_fraction

    def fit(self,x,y,sample_weight=None):
        rng=np.random.default_rng(self.seed)
        p=np.ones(len(y))/len(y) if sample_weight is None else sample_weight/np.sum(sample_weight)
        self.members=[]
        for _ in range(self.n_estimators):
            cols=np.sort(rng.choice(x.shape[1],max(1,int(x.shape[1]*self.feature_fraction)),replace=False))
            idx=rng.choice(len(y),len(y),replace=True,p=p)
            xx=np.column_stack([np.ones(len(y)),x[idx][:,cols]])
            reg=self.alpha*np.eye(xx.shape[1]); reg[0,0]=0
            coef=np.linalg.solve(xx.T@xx+reg,xx.T@y[idx])
            self.members.append((cols,coef))
        return self

    def predict(self,x):
        return np.mean([np.column_stack([np.ones(len(x)),x[:,cols]])@coef for cols,coef in self.members],axis=0)
