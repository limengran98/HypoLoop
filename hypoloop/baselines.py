"""Feedback controls that operate on the complete prediction output."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class Memory:
    """Inverse-distance interpolation of returned feedback residuals."""
    def __init__(self, neighbors=8, shrinkage=1.0):
        self.neighbors, self.shrinkage = int(neighbors), float(shrinkage)

    def fit(self, features, prediction, target):
        self.features = np.asarray(features, dtype=float)
        self.observed = np.isfinite(target)
        self.residual = np.where(self.observed, np.asarray(target)-prediction, 0.0)
        return self

    def predict(self, features, prediction):
        x, p = np.asarray(features), np.asarray(prediction)
        distance = np.linalg.norm(x[:, None]-self.features[None], axis=-1)
        neighbors = np.argsort(distance, axis=1)[:, :min(self.neighbors, len(self.features))]
        result = []
        for row, ix in zip(distance, neighbors):
            weight = (1/np.maximum(row[ix], 1e-8))[:, None]*self.observed[ix]
            correction = (weight*self.residual[ix]).sum(0)/np.maximum(weight.sum(0), 1e-12)
            result.append(correction)
        return p+self.shrinkage*np.asarray(result)


class Calib:
    """An external scale-and-offset correction with original weights fixed."""
    def fit(self, prediction, target, *, binary=False, ridge=0.01):
        p, y = np.asarray(prediction), np.asarray(target)
        observed = np.isfinite(y)
        p, y = p[observed], y[observed]
        def objective(z):
            out = (1+z[0])*p+z[1]
            if binary:
                values, grad = np.logaddexp(0, out)-y*out, expit(out)-y
            else:
                values, grad = (out-y)**2, 2*(out-y)
            return float(values.mean()+ridge*np.mean(z*z)), np.array([(grad*p).mean(), grad.mean()])+ridge*z
        self.values = minimize(objective, np.zeros(2), jac=True, method="L-BFGS-B",
                               options={"maxiter":80}).x
        return self

    def predict(self, prediction):
        return (1+self.values[0])*np.asarray(prediction)+self.values[1]
