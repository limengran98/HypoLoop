# Feedback-guided native weight editing

An editing plan contains two decisions: **Where**, the selected native parameter
coordinates, and **How**, the objective used to fit their correction.

## Where: native tensors and coordinate blocks

The model exposes its complete named-parameter inventory. A plan selects whole
tensors or contiguous blocks with selectors such as
<code>conv0.weights1::flat[1130:8747]</code>. Real and imaginary coordinates are
interleaved for complex Fourier parameters. Tied aliases refer to the same
physical parameters. The default editing budget is 8,192 real coordinates in
at most eight blocks.

The coordinate map $D_M$ places normalized changes into the selected blocks.
Each block uses its original RMS, with a minimum scale of 0.01. Coordinates
outside the chosen blocks retain their original values.

## How: executable error weighting

The fitting specification contains the loss, ridge coefficient, reference
preservation penalty, patch scale, and an executable weighting expression.
Expressions operate on fitting-fold residuals, predictions, targets, observation
ages, and condition indicators. Their normalized output defines $a_i$.

For example, DIODE uses

$$g(r)=\operatorname{clip}(\exp(2r),0.2,5), \qquad r=(f_{W_0}(x)-y)/s.$$

This directs the fit toward depths that the original model overestimates.
Weights are computed separately on each fitting fold and remain fixed during
that optimization.

## Numerical fitting and validation

The native solver optimizes

$$
\mathcal L(z)=
\sum_i a_i\,\ell(\widetilde f_{W_0+D_Mz}(x_i),\widetilde y_i)
+\frac{\gamma}{d_M}\|z\|_2^2
+\frac{\lambda}{n_R}
\|\widetilde f_{W_0+D_Mz}(X_R)-\widetilde f_{W_0}(X_R)\|_2^2 .
$$

The candidate checkpoint is $W'=W_0+\eta D_M\widehat z$.
Each candidate starts at the same original weights. The native forward keeps
the original nonlinearities and output transformations. The paper recipe
uses 80 Adam steps, learning rate 0.03, and two proposal stages. Each stage
evaluates four shared objective anchors, one proposed objective, and one
shared additional objective.

Grouped or temporal feedback folds select candidates. Deployment requires a
relative CV improvement of at least 0.001 over the original model, at most
0.05 relative degradation on accumulated feedback, and the task's native
feasibility checks. The accepted candidate replaces the current total patch.

The new model predicts the next batch and saves its checkpoint before the
next measurements are released. Its errors and completed validation records
inform the next plan.

## Implementation map

| Component | Code |
| :--- | :--- |
| Plans and fitting objectives | [plans.py](../hypoloop/plans.py), [programs.py](../hypoloop/programs.py) |
| Native coordinate selection and checkpoints | [native.py](../hypoloop/native.py) |
| True-native optimization | [solver.py](../hypoloop/solver.py) |
| Linearized proposal variants | [linearized.py](../hypoloop/linearized.py), [linear.py](../hypoloop/linear.py) |
| Validation and iterative revision | [loop.py](../hypoloop/loop.py) |
| DeepSeek and Claude requests | [transport.py](../hypoloop/transport.py) |
| Feedback controls | [baselines.py](../hypoloop/baselines.py) |
| Predict-then-reveal stream | [stream.py](../hypoloop/stream.py) |
