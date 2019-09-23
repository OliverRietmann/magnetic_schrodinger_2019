"""The WaveBlocks Project

Function for mapping complex numbers to colors specified
by the usual color map used in quantum mechanics.

@author: R. Bourquin
@copyright: Copyright (C) 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import meshgrid, angle, empty, pi, fmod, abs, arctan2, real, where
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.colors import hsv_to_rgb, ListedColormap
from matplotlib.pyplot import gca


def color_map(data, phase=None, modulus=None, darken=1.0):
    """Color mapping according to the QM standard map.

    :param data: The complex numbers.
    :param phase: The phase of the complex numbers, computed if not given.
    :param modulus: The modulus of the complex numbers, computed if not given.
    :param darken: How strong to take into account the modulus of the data to darken colors.
                   Values with :math:`|z| = R` will get fully saturated colors
                   while :math:`|z| = 0` is black and :math:`|z| \rightarrow \infty`
                   get whiter and whiter.
    :type darken: Float or ``None`` to disable darkening of colors. Default is :math:`R = 1.0`.
    :param axes: The axes instance used for plotting.
    """
    if len(data.shape) == 1:
        hsv_colors = empty((1,) + data.shape + (3,))
    else:
        hsv_colors = empty(data.shape + (3,))

    if phase is None:
        phase = angle(data)

    hsv_colors[..., 0] = 0.5 * fmod(phase + 2 * pi, 2 * pi) / pi
    hsv_colors[..., 1] = 1.0
    hsv_colors[..., 2] = 1.0

    # Darken colors such that 0+0i maps to black
    if darken is not None:
        if modulus is None:
            modulus = abs(data)

        # Lightness
        hsv_colors[..., 2] = 2 * arctan2(real(modulus), darken) / pi

        # Saturation
        l = hsv_colors[..., 2]
        hsv_colors[..., 1] = where(l <= 0.5, 2 * l, 2 * (1 - l))

    return hsv_to_rgb(hsv_colors)

"""The WaveBlocks Project

Function for plotting complex valued functions
of two real variables with the values encoded
by the usual color code.

@author: R. Bourquin
@copyright: Copyright (C) 2012, 2014, 2016 R. Bourquin
@license: Modified BSD License
"""


def plotcf2d(x, y, z, darken=None, axes=None, limits=None, **kwargs):
    r"""Plot complex valued functions :math:`\mathbb{R}^2 \rightarrow \mathbb{C}`
    with the usual color code.

    :param x: The :math:`x` values.
    :param x: The :math:`y` values.
    :param z: The values :math:`z = f(x,y)`.
    :param darken: How strong to take into account the modulus of the data to darken colors.
                   Values with :math:`|z| = R` will get fully saturated colors
                   while :math:`|z| = 0` is black and :math:`|z| \rightarrow \infty`
                   get whiter and whiter.
    :type darken: Float or ``None`` to disable darkening of colors. Default is :math:`R = 1.0`.
    :param axes: The axes instance used for plotting.
    """
    if limits is None:
        xmin = real(x).min()
        xmax = real(x).max()
        ymin = real(y).min()
        ymax = real(y).max()
        extent = [xmin, xmax, ymin, ymax]
    else:
        xmin = limits[0]
        xmax = limits[1]
        ymin = limits[2]
        ymax = limits[3]
        extent = [xmin, xmax, ymin, ymax]

    kw = {'extent': extent,
          'origin': 'lower',
          'interpolation': 'bilinear',
          'aspect': 'equal',
          'cmap': 'hsv',
          'vmin': 0,
          'vmax': 2*pi}
    kw.update(kwargs)

    # Plot to the given axis instance or retrieve the current one
    if axes is None:
        axes = gca()

    # Region to cut out
    x = x.reshape(1, -1)
    y = y.reshape(-1, 1)
    i = where((xmin <= x) & (x <= xmax))[1]
    j = where((ymin <= y) & (y <= ymax))[0]
    I, J = meshgrid(i, j)

    # Color code and plot the data
    #cmap = ListedColormap(color_map(z[J, I], darken=None))
    return axes.imshow(color_map(z[J, I], darken=darken), **kw)#, cmap
