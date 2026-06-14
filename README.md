# Tumbling Toast Simulation

A Python simulation of the experiment in M. E. Bacon, G. Heald, and M. James,
"A closer look at tumbling toast," *Am. J. Phys.* **69**(1), 38–43 (2001).

The goal is to reproduce two results from the paper: the free-fall angular
velocity as a function of overhang (Table I) and the butter-side up/down
landing behavior (Table II). The key point of the paper is that you only get
the right answer if you include the slipping stage, so the code models that
explicitly instead of assuming the board leaves the edge the instant it starts
to slide.

## The model

A rigid board of length 2a and thickness b starts horizontal on the table with
its center of mass hanging past the edge by an overhang d0. It goes through
three stages:

1. **No slip**: the board pivots about the edge. The angular acceleration
   comes from Eq. (13).
2. **Slipping**: once F_f / F_N reaches the static friction coefficient, the
   board slides while still touching the edge. Kinetic friction acts here, and
   the motion is found by solving Eqs. (4), (5), (7), and (9) together.
3. **Free fall**: when the normal force drops to zero the board leaves the
   edge and falls. The landing side is decided by the total rotation angle when
   the lower corner reaches the floor (76 cm down).

All lengths are kept in SI units inside the code; the command-line options take
cm because that is what the experiment uses.

## Default parameters (from the paper)

- board length 2a = 10.2 cm
- board thickness b = 1.3 cm
- static friction coefficient mu_s = 0.32
- kinetic friction coefficient mu_k = 0.24
- table height = 76 cm
- g = 9.8 m/s^2

All of these can be overridden from the command line (see below).

## Requirements

Python 3 with `numpy`, `scipy`, and `matplotlib`.

```
pip install numpy scipy matplotlib
```

## Running it

Sweep over a range of overhangs (this is also what runs if you give no
command):

```
python toast_simulation.py
python toast_simulation.py sweep
```

Simulate a single overhang:

```
python toast_simulation.py single --overhang-cm 1.10
```

Compare directly against the paper's Table I angular velocities and print the
RMS difference:

```
python toast_simulation.py paper
```

Change the experimental parameters, e.g. different friction and table height
with a finer overhang step:

```
python toast_simulation.py sweep --mu-s 0.35 --mu-k 0.22 --height-cm 80 --step-cm 0.02
```

If you know the board mass you can pass it to also get the forces in newtons
(otherwise only force-per-mass is reported):

```
python toast_simulation.py single --overhang-cm 1.10 --mass-g 42
```

Add `--no-plots` to any command to skip the PNG figures.

## Output

Everything is written to `results/`. The exact filenames depend on which
command you run.

**`single` command** (filenames use the overhang, with the dot written as `p`,
e.g. an overhang of 1.10 cm gives `single_1p100cm...`):

- `single_<d0>cm_summary.csv`: one row with the main numbers for that run:
  slip time/angle/angular velocity, lift-off time/angle, free-fall angular
  velocity, center-of-mass height and velocity at lift-off, total landing
  angle, and the landing side.
- `single_<d0>cm_timeseries.csv`: the full motion sampled every 1 ms across
  all three stages, including r, theta, alpha, the angular velocity, normal and
  friction force per mass (and in newtons if `--mass-g` was given), and the
  center-of-mass and lower-edge heights.
- `single_<d0>cm.png`: three stacked plots versus time: alpha, the angular
  velocity, and the lower-edge height together with the normal force, with the
  lift-off moment marked.

**`sweep` command:**

- `sweep_summary.csv`: one row per overhang over the whole sweep.
- `sweep_angular_velocity.png`: simulated free-fall angular velocity vs
  overhang, with the paper's Table I data points and error bars on top.
- `sweep_landing_angle.png`: landing angle vs overhang, with the
  butter-side-down band (90°~270°) shaded and the observed up/down points from
  Table II marked.

**`paper` command:**

- `paper_table_velocity_comparison.csv`: the simulated angular velocity next to
  the measured Table I value, the standard deviation, and the difference for
  each overhang. The console also prints the overall RMS difference.
- `paper_points_angular_velocity.png` and `paper_points_landing_angle.png`:
  the same two plots as the sweep, but only at the overhangs listed in the
  paper.

The CSV files are the easiest place to read off the actual numbers.
