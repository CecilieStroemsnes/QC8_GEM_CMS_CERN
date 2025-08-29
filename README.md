# Quality Control 8: Cosmic Ray efficiency simulation

## Layout
- `Classes/`
  - `geometry.py` — ME0_Geometry (detector + scintillator layout)
  - `simulation.py` — GEMTrajectorySimulator (+ `tally_hits_by_eta` helper)
  - `plots.py` — Plots 
  - `__init__.py` — exports the public API
- `run.py` — Script to run code
- `plots` - Folder with all outputs

## Quickstart
```bash
python run.py - 
```
This runs the default setting with the stack positioned high (close to the top scintillator).

```
python run.py --position low 
```
Here, the position is user-defined.


# ------------ Simulation Notes ------------

# Cosmic rays

For cosmic rays, we use a $cos^2(\theta)$ distribution for downward-going muons.


$\theta$ is the zenith angle ($\theta$ = 0 is straight down, and $\theta$ increases towards the horizon).

## Changing variables
To simplify, we use $\mu = cos(\theta)$, where

$\theta = 0 \Rightarrow \mu = 1$

$\theta = 90 \Rightarrow \mu = 0$

Then $d\mu = - sin(\theta) d\theta$

So the probability density function, $p(\theta) \propto cos^2(\theta) \cdot sin(\theta)$, using $cos(\theta)=\mu$ and $sin(\theta)d\theta=-d\mu$, becomes:

$p(\theta)d\theta \propto \mu^2 \cdot (-d\mu)$

$\Downarrow$

$p(\mu)d\mu \propto \mu^2 d\mu$

*(Minus just flips integration limits... ignored for probabilities)*

So now we have a simple form: 

$p(\mu) \propto \mu^2$, $\mu \in [0,1]$

## Normalizing

The probability density function, $p(\mu)\propto \mu^2$, must satisfy $\int p(x) dx = 1$ when integrating over all possible values. Since our relation is proportional to, we are missing a constant C. This constant is found by normalizing:

$\int_0^1 C \mu^2 d\mu = 1$

Where the integral is $C \cdot \frac{\mu^3}{3} |_0^1 = C \cdot \frac{1}{3}$

$\Rightarrow C = 3$

$\Rightarrow p(\mu)=3\mu^2, 0 \leq \mu \leq 1$

## Sampling using Cumulative Distribution Function

The cumulative distribution function $F(\mu)$ gives the probability that the random variable is less than or equal to $\mu$:

$F(\mu) = P (value \leq \mu) = \int_0^\mu p(t)dt$

So we integrate $p(\mu)$:
$F(/mu) = \int_0^\mu 3t^2 dt = [t^3]_0^\mu = \mu^3$

Now we set $F(\mu)=U$, so $U \sim Uniform(0,1)$

Then $\mu^3 = U \Rightarrow \mu = U^{1/3}$
