.. py:currentmodule:: lsst.gkcoadd

.. _lsst.gkcoadd:

############
lsst.gkcoadd
############

This package implements *Gaussian-Kaiser coaddition*, a variant of the optimal coaddition algorithm devised by Nick Kaiser in an unpublished ~2004 Pan-STARRS technical note, and later rediscovered by :cite:`2017ApJ...836..187Z` and :cite:`2017ApJ...836..188Z`.

.. _lsst.gkcoadd-algorithm:

Algorithm
=========

General Case
------------

Given a set of observations of the same patch of sky :math:`z_i` (2-d images, but in this notation treated as flattened vectors) with point-spread function (PSF) *convolution and reprojection operators* :math:`\phi_i` and Gaussian noise covariance matrices :math:`C_i`, we can compute the log likelihood of any above-the-atmosphere static-sky model :math:`\alpha` as

.. math::  L = \sum_i (z_i - \phi_i \alpha)^T C_i^{-1} (z_i - \phi_i \alpha)

Our goal is a coadd that is a sufficient statistic for these data, assuming this model, which we define as a single image in which the likelihood takes the same form:

.. math::  L = (z_c - \phi_c \alpha)^T C_c^{-1} (z_c - \phi_c \alpha)

If we set these equal and group by powers of :math:`\alpha` we have a system of equations we can solve for :math:`z_c`, :math:`C_c`, and :math:`\phi_c`:

.. math::

   \phi_c^T C_c^{-1} \phi_c &= \sum_i \phi_i^T C_i^{-1} \phi_i \\
   \phi_c^T C_c^{-1} z_c &= \sum_i \phi_i^T C_i^{-1} z_i \\
   z_c^T C_c^{-1} z_c &= \sum_i z_i^T C_i^{-1} z_i

The last equation does not depend on :math:`\alpha` and can be ignored.
The first two are still underdetermined, which we address with two additional constraints:

- :math:`\phi_c` can't just be any matrix; it must be a normalized convolution operator (or at least locally one, with some way to represent weak spatial variation).
- :math:`C_c` should be diagonal, or as close to diagonal as we can make it.

This is a completely general result for any static-sky likelihood, but in the general case it is computationally impractical, at least for brute-force linear algebra; the matrices are just too big.

Simple Fourier Case
-------------------

If the input PSFs are band-limited, the input noise is constant and uncorrelated, and the input images are already in the coordinate frame of the coadd, Fourier methods make this problem tractable, by diagonalizing all of the matrices.
Writing :math:`\hat z` for the DFT of an image :math:`z` and :math:`\hat{\phi}_i(k)` for the DFT of the PSF kernel,

- each convolution operator :math:`\phi_i` is diagonal in Fourier space, with eigenvalues :math:`\hat{\phi}_i(k)`;

- each covariance is a multiple of the identity, :math:`C_i = \sigma_i^2 I`;

- and the transpose of a real convolution is the convolution with the reflected kernel, whose Fourier symbol is the complex conjugate :math:`\hat{\phi}_i^*(k)`.

The first two equations then decouple into one scalar equation per Fourier mode :math:`k`:

.. math::

   \frac{\hat{\phi}_c^*(k) \hat{\phi}_c(k)}{\hat C_c(k)} &= \sum_i \frac{|\hat{\phi}_i(k)|^2}{\sigma_i^2} \equiv \hat D(k) \\[6pt]
   \frac{\hat{\phi}_c^*(k) \hat z_c(k)}{\hat C_c(k)} &= \sum_i \frac{\hat{\phi}_i^*(k) \hat z_i(k)}{\sigma_i^2} \equiv \hat s(k)

We can also require that the coadd covariance be a multiple of the identity, i.e. :math:`C_c = \sigma_c^2 I`, which breaks the remaining degeneracy and simplifies the result to

.. math::

   \hat{\phi}_c(k) = \sigma_c \sqrt{\hat D(k)}
   \qquad
   \hat z_c(k) = \frac{\sigma_c\, \hat s(k)}{\sqrt{\hat D(k)}},

with the overall gain fixed by the PSF-normalization constraint above: the coadd PSF must conserve flux, :math:`\hat{\phi}_c(0) = 1`.

It is tempting to deconvolve further by another factor of :math:`\sqrt{\hat D(k)}`, but this is unstable: it tries to recover Fourier modes that were never available in any input image.

The algorithm (Kaiser's and Zackay & Ofek's) is thus:

1. Convolve each image with the transpose of its own PSF.
2. Take the variance-weighted average of those convolved warps.
3. Deconvolve the coadd (using FFTs) by dividing by :math:`\sqrt{\hat D(k)}`, the square root of the inverse-variance-weighted sum of the input PSF power spectra (with the same overall normalization convention as step 2), leaving white noise and the combined PSF :math:`\hat{\phi}_c(k) = \sigma_c \sqrt{\hat D(k)}`.

.. TODO[algorithm]: no policy is stated yet for the division by sqrt(D) where D(k)
   becomes small near and beyond the band limit (window function, thresholding
   the combined PSF, or something else). Zackay & Ofek do not report what their
   (toy-model) implementation used, and no other extant implementation is known
   to compare against.

Warping and Gaussian PSF Approximations
---------------------------------------

In practice, the input observations start in different coordinate frames.  One straightforward fix is to use a finite sinc-like kernel (e.g. Lanczos3) to resample ("warp") them to the right grid, and then proceed with the simple Fourier algorithm.

Because the noise is not band-limited, however, it actually makes more sense to convolve each image with its PSF-transpose *before* resampling, to act as an antialiasing filter on the noise.

Furthermore, compact sinc approximations like Lanczos inevitably have power beyond the band limit themselves, and any practical resampling with one will still make the effective coadd PSF phase-dependent even if the already-convolved image (signal and noise) is now fully band-limited.
The effect on a single warp can be large (a several-percent difference in PSF size for different subpixel offsets between the input and coadd frames), but is typically naturally averaged down by coaddition, since different observations have different subpixel offsets.

This can be mitigated more rigorously (at additional computational cost) by doing a fused Gaussian convolve+warp, i.e. warping with a truncated Gaussian approximation to the PSF.
For realistic PSFs (e.g. 0.8-1.2 arcsec FWHM on 0.2 arcsec pixels), that Gaussian kernel still has some power beyond the band limit, but exponentially less than a typical compact resampling kernel.

While using a Gaussian approximation to the PSF formally breaks optimality, the signal-to-noise loss is small, and less important than the improved accuracy from antialiasing.
It also has another possibly even more important advantage: the true PSF of any observation is not actually known; all we have is a noisy model, and that makes the fact that we need to deconvolve by something derived from it problematic.
With Gaussian approximations, we know exactly what we have done to the input images and can perform that deconvolution more robustly.

We do still need to know the effective PSF on the coadd to much better than those approximations, and this means we need to propagate it separately.
Fortunately, this is easy: we can just render each input PSF model onto an image (i.e. make a synthetic isolated star), and perform the same operations to those images to coadd them (convolve+warp, then combine with the exact same weights).

This means that the original :math:`\hat{D}(k)` doesn't exist anymore; instead we need to propagate a similar quantity derived from the Gaussian kernels :math:`G_i(x)` we *actually* convolved by, to whiten the noise on the coadd:

.. math::

   \hat s_G(k) &\equiv \sum_i \frac{\hat G_i(k) \hat z_i(k)}{\sigma_i^2}
   \qquad
   \hat D_G(k) \equiv \sum_i \frac{|\hat G_i(k)|^2}{\sigma_i^2}
   \\[6pt]
   \hat z_c(k) &= \frac{\sigma_c\, \hat s_G(k)}{\sqrt{\hat D_G(k)}},
   \qquad
   \sigma_c \equiv \frac{1}{\sqrt{\hat D_G(0)}},

where :math:`\hat z_i(k)` is now the DFT of observation :math:`i`'s fused convolved-and-warped image and :math:`\hat G_i(k)` is the DFT of its Gaussian kernel :math:`G_i`.  The numerator :math:`\hat s_G` has noise power :math:`\hat D_G`, so the division leaves white noise with covariance :math:`C_c = \sigma_c^2 I`, and the gain :math:`\sigma_c` is again fixed by flux conservation, :math:`\hat\phi_c(0) = 1`.

And then we also propagate a different sum to compute the effective coadd PSF:

.. math::

   \hat\phi_c(k) = \frac{\sigma_c}{\sqrt{\hat D_G(k)}} \sum_i \frac{\hat G_i(k) \hat \phi_i(k)}{\sigma_i^2},

Note that :math:`\hat\phi_c` is not itself a propagated image but a ratio of two propagated sums, using the same :math:`\hat D_G` that whitens the data.

Non-stationary Noise
--------------------

Real images also have non-stationary noise, especially near bright objects that contribute significant Poisson noise on top of the flatter sky background Poisson noise.
It is critical not to use the per-pixel variance as the coaddition weights, as that makes it generally impossible for the coadd to have a well-defined effective PSF at all.
Instead we continue to combine images with weights derived from some central tendency of the variance over each observation.
This weakly breaks the optimality of the coadd algorithm and will leave some amount of correlated noise, but as with PSFs the most important mitigation is to propagate the true noise to the coadds.

Tracking a full per-pixel covariance is not practical, so we take two complementary approaches:

- We propagate a "variance plane" image through the full coaddition process using standard linear-operator propagation-of-uncertainty.  This does a good job of capturing the increase in noise due to bright objects, but ignores any remaining covariance.

- We propagate a very small number (2-3) of Monte Carlo noise realizations through the full coaddition process.  This is a very noisy estimate of the uncertainty, but it fully captures the covariance.

We *may* also estimate a third, a single phase-averaged covariance matrix for the coadd, if that proves useful.

Cells
-----

Realistic PSFs have spatial variation that is significant on scales larger than 100-200 pixels.
Realistic observations also have boundaries, so over any large area the set of observations contributing to the coadd changes discontinuously.

We handle both of these issues by coadding in cells, in which the PSF and the set of inputs is assumed constant.
The normalized weights are thus also exactly constant over a cell.

Naively, this will require coadding a buffer region around each cell in order to perform the deconvolution, though there may be ways to avoid this by solving a more general inverse linear problem (involving neighboring cells with different PSFs) instead of a simple FFT deconvolution.

In regions where there are relatively few observations, and hence strict cell-inclusion criteria would throw away too much precious data, we will instead relax those criteria and instead set advisory mask plane bits that indicate that the effective PSF for a cell may not be valid for certain pixels.

.. TODO[algorithm]: cell mechanics are unspecified: the buffer width required
   around each cell for the FFT deconvolution, the treatment of cell boundaries
   (FFT wraparound vs zero-padding vs the more general inverse problem sketched
   above), and the quantitative criterion for setting the advisory
   "effective PSF may not be valid" mask bits.

.. TODO[algorithm]: consider an alternative in which the coadd's effective PSF
   is smooth across cell boundaries by construction, at the cost of not fully
   decorrelating the noise: cell-boundary PSF discontinuities are plausibly a
   bigger problem for users than cell-boundary noise discontinuities, especially since the Monte Carlo noise realizations capture noise discontinuities.

.. _lsst.gkcoadd-references:

References
==========

.. bibliography::
   :cited:

.. _lsst.gkcoadd-contributing:

Contributing
============

``lsst.gkcoadd`` is developed at https://github.com/lsst-dm/gkcoadd.
You can find Jira issues for this module under the `gkcoadd <https://jira.lsstcorp.org/issues/?jql=project%20%3D%20DM%20AND%20component%20%3D%20gkcoadd>`_ component.

.. _lsst.gkcoadd-pyapi:

Python API reference
====================

.. automodapi:: lsst.gkcoadd
   :no-main-docstr:
   :no-inheritance-diagram:
