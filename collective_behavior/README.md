# Collective Behaviors

This folder contains the Python implementation of the collective behavior simulations presented in the paper and in the SI Appendix.


The folder includes multiple simulation scenarios:

- `Polarization/` – Code for the polarization experiment
- `Milling/` – Code for the milling experiment
- `Leaders/` – Code for the experiment with some agents informed about a goal destination


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
