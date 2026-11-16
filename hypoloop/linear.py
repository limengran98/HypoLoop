"""W0-linearized proposals with native validation supplied by the editor."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .programs import run_program

def fit(p0, jacobian, w0, scales, y, reference_jacobian, indices, spec, *,
        output_scale, age=None, condition=None, optimizer='lbfgs', max_iterations=80):
    """Fit a W0-anchored weighted linearized objective."""
    started=time.perf_counter()
    p0,y=np.asarray(p0,dtype=float),np.asarray(y,dtype=float)
    observed=np.isfinite(y)
    if not observed.any():raise ValueError('No observed feedback outputs')
    y=np.where(observed,y,0.)
    w0,scales=np.asarray(w0,dtype=float),np.asarray(scales,dtype=float)
    jac=np.asarray(jacobian,dtype=float);ref=np.asarray(reference_jacobian,dtype=float)
    if p0.ndim!=2 or min(p0.shape)<1 or w0.ndim!=1 or scales.shape!=w0.shape:
        raise ValueError('Invalid native head array ranks')
    n,d=p0.shape;ix=np.asarray(indices)
    if ix.ndim!=1 or not ix.size or not np.issubdtype(ix.dtype,np.integer) or np.any(ix<0) or np.any(ix>=len(w0)) or len(np.unique(ix))!=len(ix):
        raise ValueError('Scope must contain unique in-range integer indices')
    if ref.ndim!=3 or ref.shape[1:]!=(d,len(w0)) or not len(ref):
        raise ValueError('Invalid reference shape')
    if any(not np.isfinite(x).all() for x in (p0,w0,scales,jac,ref)) or np.any(scales<=0):
        raise ValueError('Native arrays must be finite and scales positive')
    if y.shape!=p0.shape or jac.shape!=(n,d,len(w0)) or not np.isfinite(y).all():
        raise ValueError('Invalid registered affine observations')
    if not np.isfinite(output_scale) or output_scale<=0:raise ValueError('Invalid base scale')
    binary=spec['loss'] in ('bce','brier')
    scale=1. if binary else output_scale
    full=(jac*scales/scale).reshape(-1,len(w0))
    a=full[:,ix];r=(ref*scales/scale).reshape(-1,len(w0))[:,ix]
    base=(p0/scale).ravel();target=(y/scale).ravel()
    if binary and (np.any(y<0) or np.any(y>1)):raise ValueError('BCE labels outside [0,1]')
    original_residual=(expit(p0)-y) if binary else (p0-y)/output_scale
    variables=dict(residual=original_residual,target=y if binary else y/output_scale,
                   prediction=expit(p0) if binary else p0/output_scale,
                   age=np.broadcast_to(np.asarray(np.zeros(n) if age is None else age)[:,None],(n,d)),
                   condition=np.broadcast_to(np.asarray(np.zeros(n) if condition is None else condition)[:,None],(n,d)))
    raw_weights=run_program(spec['weight_program'],variables,(n,d))*observed
    q=(raw_weights/raw_weights.sum()).ravel()
    beta=spec['ridge']*max(float(np.mean(np.sum(full**2,axis=0))/len(full)),1e-12)
    refgram=(r.T@r)/max(len(r),1)
    penalty=spec['preservation']*refgram+beta*np.eye(len(ix))
    cost=dict(objective_evaluations=0,optimizer_steps=0,linear_solves=0)
    def fun(z):
        cost['objective_evaluations']+=1
        pred=base+a@z;e=pred-target
        if spec['loss']=='bce':
            values=np.logaddexp(0,pred)-target*pred;g=expit(pred)-target
        elif spec['loss']=='brier':
            probability=expit(pred);error=probability-target
            values=error**2;g=2*error*probability*(1-probability)
        elif spec['loss']=='smooth_mae':
            v=np.hypot(e,.01);values=v-.01;g=e/v
        elif spec['loss']=='huber':
            ae=np.abs(e);values=np.where(ae<=1,.5*e**2,ae-.5);g=np.clip(e,-1,1)
        else:values=e**2;g=2*e
        return float(q@values+z@penalty@z), a.T@(q*g)+2*penalty@z
    initial=fun(np.zeros(len(ix)))[0]
    if spec['loss']=='mse' and optimizer!='adam':
        z=np.linalg.solve(a.T@(q[:,None]*a)+penalty,-a.T@(q*(base-target)))
        cost['linear_solves']=1;cost['optimizer_steps']=1;status='analytic'
    elif optimizer=='adam':
        z=np.zeros(len(ix));m=z.copy();v=z.copy();best=z.copy();best_loss=initial
        # Normalized-coordinate Adam.
        for t in range(1,max_iterations+1):
            value,g=fun(z)
            if value<best_loss:best_loss,best=value,z.copy()
            m=.9*m+.1*g;v=.999*v+.001*g*g
            z-=.03*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
        if fun(z)[0]<best_loss:best=z.copy()
        z=best;cost['optimizer_steps']=max_iterations;status='fixed_adam_budget'
    else:
        solved=minimize(fun,np.zeros(len(ix)),method='L-BFGS-B',jac=True,
                        options={'maxiter':max_iterations,'ftol':1e-10,'gtol':1e-7,'maxls':20})
        z=solved.x;cost['optimizer_steps']=int(solved.nit);status=str(solved.message)
    if not np.isfinite(z).all():raise FloatingPointError('Nonfinite fitted patch')
    optimized=fun(z)[0];z=z*spec['step']
    result=w0.copy();result[ix]+=scales[ix]*z
    result[ix]=result[ix].astype(np.float32).astype(np.float64)
    final_z=(result[ix]-w0[ix])/scales[ix]
    final=fun(final_z)[0]
    return result,dict(status=status,objective_before=initial,objective_optimized=optimized,
        objective_after_native_rounding=final,fixed_beta=beta,weight_min=float(raw_weights.min()),
        weight_max=float(raw_weights.max()),native_changed_parameters=int(np.count_nonzero(result.astype(np.float32)!=w0.astype(np.float32))),
        total_patch_scaled_rms=float(np.sqrt(np.mean(((result-w0)/scales)**2))),
        cost={**cost,'seconds':time.perf_counter()-started},optimizer=optimizer,
        program_training_fold_only=True,actual_native_dtype='float32')
