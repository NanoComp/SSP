# Smoothed Subpixel Projection (SSP)

This is a repository for code implementing the **smoothed subpixel projection (SSP)** scheme for topology optimization, which blends density-based and level-set methods in a differentiable way, along with related techniques, as described in the papers:

* A. M. Hammond, A. Oskooi, I. M. Hammond, M. Chen, S. E. Ralph, and S. G. Johnson, [“Unifying and accelerating level-set and density-based topology optimization by subpixel-smoothed projection,”](http://doi.org/10.1364/OE.563512) Optics Express, vol. 33, pp. 33620–33642, July 2025. Editor's Pick.
* G. Romano, R. Arrieta, and S. G. Johnson, [“Differentiating through binarized topology changes: Second-order subpixel-smoothed projection,”](http://arxiv.org/abs/2601.10737) arXiv.org e-Print archive, 2601.10737, January 2026.
* R. Arrieta, G. Romano, and S. G. Johnson, [“Hyperparameter-free minimum-lengthscale constraints for topology optimization,”](http://arxiv.org/abs/2507.16108) arXiv.org e-Print archive, 2507.16108, July 2025.

## Installation

Install the PyPI distribution:

```bash
pip install ssp-topopt
```

The Python import package is `ssp_topopt`:

```python
from ssp_topopt import conic_filter, get_conic_radius_from_eta_e, ssp1_bilinear
```

For local development:

```bash
python -m pip install -e ".[dev]"
```
