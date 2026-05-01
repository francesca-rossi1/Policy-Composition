# Policy Composition for Emerging Flocking Behaviors

This folder contains the Python implementation of the collective behavior simulations presented in the paper and in the SI Appendix.


The folder includes multiple simulation scenarios:

- `Polarization/` – Code for the polarization experiment
- `Milling/` – Code for the milling experiment
- `Leaders/` – Code for the experiment with some agents informed about a goal destination
- `Varying_epsilon/` - Code for analyzing how varying the regularization parameter influences the collective behavior in the experiments with some agents informed about a goal destination
- `Different_models/` – Code for the experiments with some agents informed about a goal destination implementing our policy composition model, and other uninformed agents implementing three different models from the literature

See the 'README.md' inside each folder for details.


## Requirements

The simulations and plotting scripts rely on the following Python packages:

- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`
- `python-ternary`
- `pickle`

You can install the required Python packages with:

```bash
pip install -r requirements.txt
```

### Additional requirement

- `ffmpeg` is required for video generation (used by `matplotlib.animation.FFMpegWriter`)
