import numpy as np
from .disc2d import InputError, PlotTools, Tools, path_icons
from radio_beam import Beam
from astropy.convolution import Gaussian2DKernel
from astropy import units as u
from astropy import constants as apc
from astropy.io import fits
from astropy.wcs import utils as aputils, WCS
import warnings
import os
import copy
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Cursor, Slider, RectangleSelector
        
SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 22


class Cube(object):
    def __init__(self, data, header, vchannels, wcs, beam):
        self.data = data
        self.header = header
        self.vchannels = vchannels
        self.wcs = wcs
        # Assuming (nchan, nx, nx), nchan should be equal to cube_vel.spectral_axis.size
        self.nchan, self.nx, _ = np.shape(data)

        if isinstance(beam, Beam):
            self._init_beam_kernel() # Get 2D Gaussian kernel from beam
        elif beam is None:
            pass
        else:
            raise InputError(beam, "beam must either be None or radio_beam.Beam object")
        self.beam = beam

        self._interactive = self._cursor
        self._interactive_path = self._curve

    def _init_beam_kernel(self):
        """
        Compute 2D Gaussian kernel in pixels from beam info.

        """
        sigma2fwhm = np.sqrt(8 * np.log(2))
        # pixel size in CUNIT2 units
        pix_scale = np.abs(self.header["CDELT2"]) * u.Unit(self.header["CUNIT2"])
        x_stddev = ((self.beam.major / pix_scale) / sigma2fwhm).decompose().value
        y_stddev = ((self.beam.minor / pix_scale) / sigma2fwhm).decompose().value
        beam_angle = (90 * u.deg + self.beam.pa).to(u.radian).value
        self.beam_kernel = Gaussian2DKernel(x_stddev, y_stddev, beam_angle)

    def _writefits(self, logkeys=None, tag="", **kwargs):
        """
        Write fits file
            
        Paramaters
        ----------
        logkeys : list of str, optional
            List of keys to append in the output file name. If multiple keys are pased the order in the output name is maintained.
            Only the keys present in the cube header will be added to the output file name.

        tag : str, optional
            String to add at the end of the output filename.

        kwargs : keyword arguments
            Additional keyword arguments to pass to `~astropy.io.fits.writeto` function.
           
        """
        ktag = ""
        if logkeys is not None:
            for key in logkeys:
                if key in self.header and self.header[key]:
                    ktag += "_" + key.lower()

        self.fileroot += ktag + tag
        fits.writeto(self.fileroot + ".fits", self.data, header=self.header, **kwargs)

    def convert_to_tb(self, planck=True, writefits=True, tag="", **kwargs
    ):  
        """
        Convert intensity to brightness temperature in units of Kelvin.

        Parameters
        ----------
        planck : bool, optional
            If True, it uses the full Planck law to make the conversion, else it uses the Rayleigh-Jeans approximation. 

        writefits : bool, optional
            If True it creates a FITS file with the new intensity data and header.
        
        tag : str, optional
            String to add at the end of the output filename.

        kwargs : keyword arguments
            Additional keyword arguments for `~astropy.io.fits.writeto` function.
           
        """
        hdrkey = "CONVTB"
        hdrcard = "Converted to Tb by DISCMINER"
        kwargs_io = dict(overwrite=True)  # Default kwargs
        kwargs_io.update(kwargs)

        I = self.data * u.Unit(self.header["BUNIT"]).to("beam-1 Jy")
        nu = self.header["RESTFRQ"]  # in Hz
        bmaj = self.beam.major.to(u.arcsecond).value
        bmin = self.beam.minor.to(u.arcsecond).value
        # area of gaussian beam
        beam_area = u.au.to("m") ** 2 * np.pi * (bmaj * bmin) / (4 * np.log(2))
        # beam solid angle: beam_area/(dist*pc)**2.
        #  dist**2 cancels out with beamarea's dist**2 from conversion of bmaj, bmin to mks units.
        beam_solid = beam_area / u.pc.to("m") ** 2
        Jy_to_SI = 1e-26
        c_h = apc.h.value
        c_c = apc.c.value
        c_k_B = apc.k_B.value

        if planck:
            Tb = (
                np.sign(I)
                * (
                    np.log(
                        (2 * c_h * nu ** 3)
                        / (c_c ** 2 * np.abs(I) * Jy_to_SI / beam_solid)
                        + 1
                    )
                )
                ** -1
                * c_h
                * nu
                / (c_k_B)
            )
        else:
            wl = c_c / nu
            Tb = 0.5 * wl ** 2 * I * Jy_to_SI / (beam_solid * c_k_B)

        self.data = Tb
        self.header["BUNIT"] = "K"

        self.wcs = WCS(self.header)
        self.header[hdrkey] = (True, hdrcard)
        if writefits:
            self._writefits(logkeys=[hdrkey], tag=tag, **kwargs_io)

    def downsample(
        self, npix, method=np.median, kwargs_method={}, writefits=True, tag="", **kwargs
    ):
        """
        Downsample data cube to reduce spatial correlations between pixels and/or to save computational costs in the modelling. 

        Parameters
        ----------
        npix : int
            Number of pixels to downsample. For example, if npix=3, the downsampled-pixel will have an area of 3x3 original-pixels

        method : func, optional
            function to compute downsampling

        kwargs_method : keyword arguments
            Additional keyword arguments to pass to the input ``method``.

        writefits : bool, optional
            If True it creates a FITS file with the new intensity data and header.
        
        tag : str, optional
            String to add at the end of the output filename.

        kwargs : keyword arguments
            Additional keyword arguments to pass to `~astropy.io.fits.writeto` function.
           
        """
        hdrkey = "DOWNSAMP"
        hdrcard = "Downsampled by DISCMINER"
        kwargs_io = dict(overwrite=True)  # Default kwargs
        kwargs_io.update(kwargs)

        nchan, nx0 = self.nchan, self.nx
        nx = int(round(nx0 / npix))

        if npix > 1:
            av_data = np.zeros((nchan, nx, nx))  # assuming ny = nx
            progress = Tools._progress_bar
            di = npix
            dj = npix
            print("Averaging %dx%d pixels from data cube..." % (di, dj))
            for k in range(nchan):
                progress(int(100 * k / nchan))
                for i in range(nx):
                    for j in range(nx):
                        av_data[k, j, i] = method(
                            self.data[k, j * dj : j * dj + dj, i * di : i * di + di],
                            **kwargs_method
                        )
            progress(100)

            self.nx = nx
            self.data = av_data

            # nf: number of pix between centre of first pix in the original img and centre of first downsampled pix
            if npix % 2:  # if odd
                nf = (npix - 1) / 2.0
            else:
                nf = 0.5 + (npix / 2 - 1)

            # will be the new CRPIX1 and CRPIX2 (origin is 1,1, not 0,0)
            refpix = 1.0
            # coords of reference pixel, using old pixels info
            refpixval = aputils.pixel_to_skycoord(nf, nf, self.wcs)

            CDELT1, CDELT2 = self.header["CDELT1"], self.header["CDELT2"]
            # equivalent to CRVAL1 - CDELT1 * (CRPIX1 - 1 - nf) but using right projection
            self.header["CRVAL1"] = refpixval.ra.value
            self.header["CRVAL2"] = refpixval.dec.value
            self.header["CDELT1"] = CDELT1 * npix
            self.header["CDELT2"] = CDELT2 * npix
            self.header["CRPIX1"] = refpix
            self.header["CRPIX2"] = refpix
            self.header["NAXIS1"] = nx
            self.header["NAXIS2"] = nx

            self.wcs = WCS(self.header)
            self._init_beam_kernel()
            # keeping track of changes to original cube
            self.header[hdrkey] = (True, hdrcard)
            if writefits:
                self._writefits(logkeys=[hdrkey], tag=tag, **kwargs_io)

        else:
            print("npix is <= 1, no average was performed...")

    def clip(
        self,
        npix=0,
        icenter=None,
        jcenter=None,
        channels={"interval": None, "indices": None},
        writefits=True,
        tag="",
        **kwargs
    ):
        """
        Clip spatial and/or velocity axes of the data cube. The extent of the clipped region would be 
        ``[icenter-npix, icenter+npix]`` along the first spatial axis (normally RA), and ``[jcenter-npix, jcenter+npix]`` along the second spatial axis (normally DEC).
        
        See the description of the argument ``channels`` below for details on how the velocity axis is clipped.

        Parameters
        ----------
        npix : int
            Number of pixels to clip above and below (and to the left and right of) the reference centre of the data (icenter, jcenter). 
            The total number of pixels after clipping would be 2*npix on each spatial axis.
        
        icenter, jcenter : int, optional
            Reference centre for the clipped window. Must be integers referred to pixel ids from the input data. 
            If None, the reference centre is determined from the input header as ``icenter=int(header['CRPIX1'])`` and ``jcenter=int(header['CRPIX2'])``
        
        channels : {"interval" : [i0, i1]} or {"indices" : [i0, i1,..., in]}, optional
            Dictionary of indices to clip velocity channels from data. If both entries are None, all velocity channels are considered.         
 
            * If 'interval' is defined, velocity channels between *i0* and *i1* indices are considered, *i1* inclusive.          
            * If 'indices' is defined, only velocity channels corresponding to the input indices will be considered.            
            * If both entries are set, only 'interval' will be taken into account.
        
        writefits : bool, optional
            If True it creates a FITS file with the new intensity data and header.

        tag : str, optional
            String to add at the end of the output filename.

        kwargs : keyword arguments
            Additional keyword arguments to pass to `~astropy.io.fits.writeto` function.
        
        """
        hdrkey = "CLIPPED"
        hdrcard = "Clipped by DISCMINER"
        kwargs_io = dict(overwrite=True)  # Default kwargs
        kwargs_io.update(kwargs)

        if icenter is not None:
            icenter = int(icenter)
        else:  # Assume reference centre at the centre of the image
            icenter = int(0.5 * self.header["NAXIS1"] + 1)
        if jcenter is not None:
            jcenter = int(jcenter)
        else:
            jcenter = int(0.5 * self.header["NAXIS2"] + 1)

        if channels is None:
            channels = {}
        if "interval" not in channels.keys():
            channels.update({"interval": None})
        if "indices" not in channels.keys():
            channels.update({"indices": None})
        if channels["interval"] is not None:
            i0, i1 = channels["interval"]
            idchan = np.arange(i0, i1 + 1).astype(int)
        elif channels["indices"] is not None:
            idchan = np.asarray(channels["indices"]).astype(int)
            warnings.warn(
                "Note that if you select channels that are not regularly spaced the header of the output fits file will not reflect this information and therefore external analysis tools such as CASA or DS9 will not display the velocity information correctly.",
            )
        else:
            idchan = slice(None)

        self.data = self.data[idchan]
        self.vchannels = self.vchannels[idchan]

        # data shape: (NAXIS3, NAXIS2, NAXIS1)
        if npix > 0:
            self.data = self.data[
                :, jcenter - npix : jcenter + npix, icenter - npix : icenter + npix
            ]
            # The following is wrong because the projection is not Cartesian:
            #  self.header["CRVAL1"] = CRVAL1 + (icenter - CRPIX1) * CDELT1.
            #   A proper conversion using wcs must be done:
            newcr = aputils.pixel_to_skycoord(icenter, jcenter, self.wcs)
            self.header["CRVAL1"] = newcr.ra.value
            self.header["CRVAL2"] = newcr.dec.value
            self.header["CRPIX1"] = npix + 1.0
            self.header["CRPIX2"] = npix + 1.0

        self.nchan, self.nx, _ = self.data.shape
        self.header["NAXIS1"] = self.nx
        self.header["NAXIS2"] = self.nx
        self.header["NAXIS3"] = self.nchan
        self.header["CRPIX3"] = 1.0
        self.header["CRVAL3"] = self.vchannels[0]
        self.header["CDELT3"] = self.vchannels[1] - self.vchannels[0]

        self.wcs = WCS(self.header)
        self.header[hdrkey] = (True, hdrcard)
        if writefits:
            self._writefits(logkeys=[hdrkey], tag=tag, **kwargs_io)

    # *********************************
    # FUNCTIONS FOR INTERACTIVE WINDOWS
    # *********************************
    @property
    def interactive(self):
        return self._interactive

    @interactive.setter
    def interactive(self, func):
        print("Setting interactive function to", func)
        self._interactive = func

    @property
    def interactive_path(self):
        return self._interactive_path

    @interactive_path.setter
    def interactive_path(self, func):
        print("Setting interactive_path function to", func)
        self._interactive_path = func

    def _surface(self, ax, *args, **kwargs):
        return Contours.emission_surface(ax, *args, **kwargs)
        
    def _plot_beam(self, ax):
        x_fwhm = self.beam_kernel.model.x_fwhm
        y_fwhm = self.beam_kernel.model.y_fwhm
        ny_pix, nx_pix = np.shape(self.data[0])
        ellipse = patches.Ellipse(
            xy=(0.05, 0.05),
            angle=90 + self.beam.pa.value,
            width=x_fwhm / nx_pix,
            height=y_fwhm / ny_pix,
            lw=1,
            fill=True,
            fc="gray",
            ec="k",
            transform=ax.transAxes,
        )
        ax.add_artist(ellipse)

    def _check_cubes_shape(self, compare_cubes):
        for cube in compare_cubes:
            if cube.data.shape != self.data.shape:
                raise InputError(compare_cubes, "Input cubes for comparison must have the same shape")

    # *************************************
    # SHOW SPECTRUM ON PIXEL and WITHIN BOX
    # *************************************    
    def _plot_spectrum_box(
        self,
        x0,
        x1,
        y0,
        y1,
        ax,
        extent=None,
        compare_cubes=[],
        stat_func=np.mean,
        **kwargs
    ):
        kwargs_spec = dict(where="mid", linewidth=2.5, label=r"x0:%d,x1:%d" % (x0, x1))
        kwargs_spec.update(kwargs)
        v0, v1 = self.vchannels[0], self.vchannels[-1]

        def get_ji(x, y):
            pass

        if extent is None:
            j0, i0 = int(x0), int(y0)
            j1, i1 = int(x1), int(y1)
        else:
            nz, ny, nx = np.shape(self.data)
            dx = extent[1] - extent[0]
            dy = extent[3] - extent[2]
            j0 = int(nx * (x0 - extent[0]) / dx)
            i0 = int(ny * (y0 - extent[2]) / dy)
            j1 = int(nx * (x1 - extent[0]) / dx)
            i1 = int(ny * (y1 - extent[2]) / dy)

        slice_cube = self.data[:, i0:i1, j0:j1]
        spectrum = np.array([stat_func(chan) for chan in slice_cube])
        ncubes = len(compare_cubes)
        if ncubes > 0:
            slice_comp = [compare_cubes[i].data[:, i0:i1, j0:j1] for i in range(ncubes)]
            cubes_spec = [
                np.array([stat_func(chan) for chan in slice_comp[i]])
                for i in range(ncubes)
            ]

        if np.logical_or(np.isinf(spectrum), np.isnan(spectrum)).all():
            return False
        else:
            plot_spec = ax.step(self.vchannels, spectrum, **kwargs_spec)
            if ncubes > 0:
                alpha = 0.2
                dalpha = -alpha / ncubes
                for i in range(ncubes):
                    ax.fill_between(
                        self.vchannels,
                        cubes_spec[i],
                        color=plot_spec[0].get_color(),
                        step="mid",
                        alpha=alpha,
                    )
                    alpha += dalpha
            else:
                ax.fill_between(
                    self.vchannels,
                    spectrum,
                    color=plot_spec[0].get_color(),
                    step="mid",
                    alpha=0.2,
                )
            return plot_spec

    def _box(self, fig, ax, extent=None, compare_cubes=[], stat_func=np.mean, **kwargs):

        def onselect(eclick, erelease):
            if eclick.inaxes is ax[0]:
                plot_spec = self._plot_spectrum_box(
                    eclick.xdata,
                    erelease.xdata,
                    eclick.ydata,
                    erelease.ydata,
                    ax[1],
                    extent=extent,
                    compare_cubes=compare_cubes,
                    stat_func=stat_func,
                    **kwargs
                )

                if plot_spec:
                    print("startposition: (%f, %f)" % (eclick.xdata, eclick.ydata))
                    print("endposition  : (%f, %f)" % (erelease.xdata, erelease.ydata))
                    print("used button  : ", eclick.button)
                    xc, yc = eclick.xdata, eclick.ydata  # Left, bottom corner
                    dx, dy = (
                        erelease.xdata - eclick.xdata,
                        erelease.ydata - eclick.ydata,
                    )
                    rect = patches.Rectangle(
                        (xc, yc),
                        dx,
                        dy,
                        lw=2,
                        edgecolor=plot_spec[0].get_color(),
                        facecolor="none",
                    )
                    ax[0].add_patch(rect)
                    ax[1].legend()
                    fig.canvas.draw()
                    fig.canvas.flush_events()

        def toggle_selector(event):
            print("Key pressed.")
            if event.key in ["C", "c"] and toggle_selector.RS.active:
                print("RectangleSelector deactivated.")
                toggle_selector.RS.set_active(False)
            if event.key in ["A", "a"] and not toggle_selector.RS.active:
                print("RectangleSelector activated.")
                toggle_selector.RS.set_active(True)

        rectprops = dict(facecolor="0.7", edgecolor="k", alpha=0.3, fill=True)
        lineprops = dict(color="white", linestyle="-", linewidth=3, alpha=0.8)

        toggle_selector.RS = RectangleSelector(
            ax[0], onselect, drawtype="box", rectprops=rectprops, lineprops=lineprops, button=[1]
        )
        cid = fig.canvas.mpl_connect("key_press_event", toggle_selector)
        return toggle_selector.RS


    def _plot_spectrum_cursor(self, x, y, ax, extent=None, compare_cubes=[], **kwargs):
        kwargs_spec = dict(where="mid", linewidth=2.5, label=r"%d,%d" % (x, y))
        kwargs_spec.update(kwargs)

        def get_ji(x, y):
            pass

        if extent is None:
            j, i = int(x), int(y)
        else:
            nz, ny, nx = np.shape(self.data)
            dx = extent[1] - extent[0]
            dy = extent[3] - extent[2]
            j = int(nx * (x - extent[0]) / dx)
            i = int(ny * (y - extent[2]) / dy)

        spectrum = self.data[:, i, j]
        v0, v1 = self.vchannels[0], self.vchannels[-1]

        if np.logical_or(np.isinf(spectrum), np.isnan(spectrum)).all():
            return False
        else:
            # plot_fill = ax.fill_between(self.vchannels, spectrum, alpha=0.1)
            plot_spec = ax.step(self.vchannels, spectrum, **kwargs_spec)
            ncubes = len(compare_cubes)
            if ncubes > 0:
                alpha = 0.2
                dalpha = -alpha / ncubes
                for cube in compare_cubes:
                    ax.fill_between(
                        self.vchannels,
                        cube.data[:, i, j],
                        color=plot_spec[0].get_color(),
                        step="mid",
                        alpha=alpha,
                    )
                    alpha += dalpha
            else:
                ax.fill_between(
                    self.vchannels,
                    spectrum,
                    color=plot_spec[0].get_color(),
                    step="mid",
                    alpha=0.2,
                )
            return plot_spec

    def _cursor(self, fig, ax, extent=None, compare_cubes=[], **kwargs):
        def onclick(event):
            if event.button == 3:
                print("Right click. Disconnecting click event...")
                fig.canvas.mpl_disconnect(cid)
            elif event.inaxes is ax[0]:
                plot_spec = self._plot_spectrum_cursor(
                    event.xdata,
                    event.ydata,
                    ax[1],
                    extent=extent,
                    compare_cubes=compare_cubes,
                    **kwargs
                )
                if plot_spec is not None:
                    print(
                        "%s click: button=%d, xdata=%f, ydata=%f"
                        % (
                            "double" if event.dblclick else "single",
                            event.button,
                            event.xdata,
                            event.ydata,
                        )
                    )
                    ax[0].scatter(
                        event.xdata,
                        event.ydata,
                        marker="D",
                        s=50,
                        facecolor=plot_spec[0].get_color(),
                        edgecolor="k",
                    )
                    ax[1].legend(
                        frameon=False, handlelength=0.7, fontsize=MEDIUM_SIZE - 1
                    )
                    fig.canvas.draw()
                    fig.canvas.flush_events()

        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        return cid

    def show(
        self,
        extent=None,
        chan_init=0,
        compare_cubes=[],
        cursor_grid=True,
        cmap="gnuplot2_r",
        int_unit=r"Intensity [mJy beam$^{-1}$]",
        pos_unit="Offset [au]",
        vel_unit=r"km s$^{-1}$",
        show_beam=False,
        surface={"args": (), "kwargs": {}},
        **kwargs
    ):

        self._check_cubes_shape(compare_cubes)        
        v0, v1 = self.vchannels[0], self.vchannels[-1]
        dv = v1 - v0
        fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
        plt.subplots_adjust(wspace=0.25)

        y0, y1 = ax[1].get_position().y0, ax[1].get_position().y1
        axcbar = plt.axes([0.47, y0, 0.03, y1 - y0])
        max_data = np.nanmax([self.data] + [comp.data for comp in compare_cubes])
        ax[0].set_xlabel(pos_unit)
        ax[0].set_ylabel(pos_unit)
        ax[1].set_xlabel("l.o.s velocity [%s]" % vel_unit)
        PlotTools.mod_major_ticks(ax[0], axis="both", nbins=5)
        ax[0].tick_params(direction="out")
        ax[1].tick_params(direction="in", right=True, labelright=False, labelleft=False)
        axcbar.tick_params(direction="out")
        ax[1].set_ylabel(int_unit, labelpad=15)
        ax[1].yaxis.set_label_position("right")
        ax[1].set_xlim(v0 - 0.1, v1 + 0.1)
        vmin, vmax = -1 * max_data / 100, 0.7 * max_data  # 0.8*max_data#
        ax[1].set_ylim(vmin, vmax)
        # ax[1].grid(lw=1.5, ls=':')
        cmap = copy.copy(plt.get_cmap(cmap))
        cmap.set_bad(color=(0.9, 0.9, 0.9))

        if show_beam and self.beam_kernel:
            self._plot_beam(ax[0])

        img = ax[0].imshow(
            self.data[chan_init],
            cmap=cmap,
            extent=extent,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        cbar = plt.colorbar(img, cax=axcbar)
        img.cmap.set_under("w")
        current_chan = ax[1].axvline(
            self.vchannels[chan_init], color="black", lw=2, ls="--"
        )
        text_chan = ax[1].text(
            (self.vchannels[chan_init] - v0) / dv,
            1.02,  # Converting xdata coords to Axes coords
            "%4.1f %s" % (self.vchannels[chan_init], vel_unit),
            ha="center",
            color="black",
            transform=ax[1].transAxes,
        )

        if cursor_grid:
            cg = Cursor(ax[0], useblit=True, color="lime", linewidth=1.5)

        def get_interactive(func):
            return func(fig, ax, extent=extent, compare_cubes=compare_cubes, **kwargs)

        interactive_obj = [get_interactive(self.interactive)]

        # ***************
        # SLIDERS
        # ***************
        def update_chan(val):
            chan = int(val)
            vchan = self.vchannels[chan]
            img.set_data(self.data[chan])
            current_chan.set_xdata(vchan)
            text_chan.set_x((vchan - v0) / dv)
            text_chan.set_text("%4.1f %s" % (vchan, vel_unit))
            fig.canvas.draw_idle()

        def update_cubes(val):
            i = int(slider_cubes.val)
            chan = int(slider_chan.val)
            vchan = self.vchannels[chan]
            if i == 0:
                img.set_data(self.data[chan])
            else:
                img.set_data(compare_cubes[i - 1].data[chan])
            current_chan.set_xdata(vchan)
            text_chan.set_x((vchan - v0) / dv)
            text_chan.set_text("%4.1f km/s" % vchan)
            fig.canvas.draw_idle()

        ncubes = len(compare_cubes)
        if ncubes > 0:
            axcubes = plt.axes([0.2, 0.90, 0.24, 0.025], facecolor="0.7")
            axchan = plt.axes([0.2, 0.95, 0.24, 0.025], facecolor="0.7")
            slider_cubes = Slider(
                axcubes,
                "Cube id",
                0,
                ncubes,
                valstep=1,
                valinit=0,
                valfmt="%1d",
                color="dodgerblue",
            )
            slider_chan = Slider(
                axchan,
                "Channel",
                0,
                self.nchan - 1,
                valstep=1,
                valinit=chan_init,
                valfmt="%2d",
                color="dodgerblue",
            )
            slider_cubes.on_changed(update_cubes) 
            slider_chan.on_changed(update_cubes) # update_cubes works for both
        else:
            axchan = plt.axes([0.2, 0.9, 0.24, 0.05], facecolor="0.7")
            slider_chan = Slider(
                axchan,
                "Channel",
                0,
                self.nchan - 1,
                valstep=1,
                valinit=chan_init,
                valfmt="%2d",
                color="dodgerblue",
            )
            slider_chan.on_changed(update_chan)

        # *************
        # BUTTONS
        # *************
        def go2cursor(event):
            # If already cursor, return 0 and pass, else, exec cursor func
            if self.interactive == self._cursor: 
                return 0
            interactive_obj[0].set_active(False)
            self.interactive = self._cursor
            interactive_obj[0] = get_interactive(self._cursor)

        def go2box(event):
            if self.interactive == self._box:
                return 0
            fig.canvas.mpl_disconnect(interactive_obj[0])
            self.interactive = self._box
            interactive_obj[0] = get_interactive(self._box)

        def go2trash(event):
            print("Cleaning interactive figure...")
            plt.close()
            chan = int(slider_chan.val)
            self.show(
                extent=extent,
                chan_init=chan,
                compare_cubes=compare_cubes,
                cursor_grid=cursor_grid,
                int_unit=int_unit,
                pos_unit=pos_unit,
                vel_unit=vel_unit,
                surface=surface,
                show_beam=show_beam,
                **kwargs
            )

        def go2surface(event):
            self._surface(ax[0], *surface["args"], **surface["kwargs"])
            fig.canvas.draw()
            fig.canvas.flush_events()

        box_img = plt.imread(path_icons + "button_box.png")
        cursor_img = plt.imread(path_icons + "button_cursor.jpeg")
        trash_img = plt.imread(path_icons + "button_trash.jpg")
        surface_img = plt.imread(path_icons + "button_surface.png")
        axbcursor = plt.axes([0.05, 0.779, 0.05, 0.05])
        axbbox = plt.axes([0.05, 0.72, 0.05, 0.05])
        axbtrash = plt.axes([0.05, 0.661, 0.05, 0.05], frameon=True, aspect="equal")
        bcursor = Button(axbcursor, "", image=cursor_img)
        bcursor.on_clicked(go2cursor)
        bbox = Button(axbbox, "", image=box_img)
        bbox.on_clicked(go2box)
        btrash = Button(axbtrash, "", image=trash_img, color="white", hovercolor="lime")
        btrash.on_clicked(go2trash)

        if len(surface["args"]) > 0:
            axbsurf = plt.axes([0.005, 0.759, 0.07, 0.07], frameon=True, aspect="equal")
            bsurf = Button(axbsurf, "", image=surface_img)
            bsurf.on_clicked(go2surface)

        plt.show(block=True)

        
    def show_side_by_side(
        self,
        cube1,
        extent=None,
        chan_init=0,
        cursor_grid=True,
        cmap="gnuplot2_r",
        int_unit=r"Intensity [mJy beam$^{-1}$]",
        pos_unit="Offset [au]",
        vel_unit=r"km s$^{-1}$",
        show_beam=False,
        surface={"args": (), "kwargs": {}},
        **kwargs
    ):

        compare_cubes = [cube1]
        self._check_cubes_shape(compare_cubes)
        
        v0, v1 = self.vchannels[0], self.vchannels[-1]
        dv = v1 - v0
        fig, ax = plt.subplots(ncols=3, figsize=(17, 5))
        plt.subplots_adjust(wspace=0.25)

        y0, y1 = ax[2].get_position().y0, ax[2].get_position().y1
        axcbar = plt.axes([0.63, y0, 0.015, y1 - y0])
        max_data = np.nanmax([self.data] + [comp.data for comp in compare_cubes])
        ax[0].set_xlabel(pos_unit)
        ax[0].set_ylabel(pos_unit)
        ax[2].set_xlabel("l.o.s velocity [%s]" % vel_unit)
        PlotTools.mod_major_ticks(ax[0], axis="both", nbins=5)
        ax[0].tick_params(direction="out")
        ax[2].tick_params(direction="in", right=True, labelright=False, labelleft=False)
        axcbar.tick_params(direction="out")
        ax[2].set_ylabel(int_unit, labelpad=15)
        ax[2].yaxis.set_label_position("right")
        ax[2].set_xlim(v0 - 0.1, v1 + 0.1)
        vmin, vmax = -1 * max_data / 100, 0.7 * max_data  # 0.8*max_data#
        ax[2].set_ylim(vmin, vmax)
        cmap = copy.copy(plt.get_cmap(cmap))
        cmap.set_bad(color=(0.9, 0.9, 0.9))

        if show_beam and self.beam_kernel:
            self._plot_beam(ax[0])

        img = ax[0].imshow(
            self.data[chan_init],
            cmap=cmap,
            extent=extent,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        img1 = ax[1].imshow(
            cube1.data[chan_init],
            cmap=cmap,
            extent=extent,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        cbar = plt.colorbar(img, cax=axcbar)
        img.cmap.set_under("w")
        img1.cmap.set_under("w")
        current_chan = ax[2].axvline(
            self.vchannels[chan_init], color="black", lw=2, ls="--"
        )
        text_chan = ax[2].text(
            (self.vchannels[chan_init] - v0) / dv,
            1.02,  # Converting xdata coords to Axes coords
            "%4.1f %s" % (self.vchannels[chan_init], vel_unit),
            ha="center",
            color="black",
            transform=ax[2].transAxes,
        )

        if cursor_grid:
            cg = Cursor(ax[0], useblit=True, color="lime", linewidth=1.5)

        def get_interactive(func):
            return func(
                fig,
                [ax[0], ax[2]],
                extent=extent,
                compare_cubes=compare_cubes,
                **kwargs
            )

        interactive_obj = [get_interactive(self.interactive)]

        # ***************
        # SLIDERS
        # ***************
        def update_chan(val):
            chan = int(val)
            vchan = self.vchannels[chan]
            img.set_data(self.data[chan])
            img1.set_data(cube1.data[chan])
            current_chan.set_xdata(vchan)
            text_chan.set_x((vchan - v0) / dv)
            text_chan.set_text("%4.1f %s" % (vchan, vel_unit))
            fig.canvas.draw_idle()

        ncubes = len(compare_cubes)
        axchan = plt.axes([0.2, 0.9, 0.24, 0.05], facecolor="0.7")
        slider_chan = Slider(
            axchan,
            "Channel",
            0,
            self.nchan - 1,
            valstep=1,
            valinit=chan_init,
            valfmt="%2d",
            color="dodgerblue",
        )
        slider_chan.on_changed(update_chan)

        # *************
        # BUTTONS
        # *************
        def go2cursor(event):
            if self.interactive == self._cursor:
                return 0
            interactive_obj[0].set_active(False)
            self.interactive = self._cursor
            interactive_obj[0] = get_interactive(self.interactive)

        def go2box(event):
            if self.interactive == self._box:
                return 0
            fig.canvas.mpl_disconnect(interactive_obj[0])
            self.interactive = self._box
            interactive_obj[0] = get_interactive(self.interactive)

        def go2trash(event):
            print("Cleaning interactive figure...")
            plt.close()
            chan = int(slider_chan.val)
            self.show_side_by_side(
                cube1,
                extent=extent,
                chan_init=chan,
                cursor_grid=cursor_grid,
                int_unit=int_unit,
                pos_unit=pos_unit,
                vel_unit=vel_unit,
                surface=surface,
                show_beam=show_beam,
                **kwargs
            )

        def go2surface(event):
            self._surface(ax[0], *surface["args"], **surface["kwargs"])
            self._surface(ax[1], *surface["args"], **surface["kwargs"])
            fig.canvas.draw()
            fig.canvas.flush_events()

        box_img = plt.imread(path_icons + "button_box.png")
        cursor_img = plt.imread(path_icons + "button_cursor.jpeg")
        trash_img = plt.imread(path_icons + "button_trash.jpg")
        surface_img = plt.imread(path_icons + "button_surface.png")
        axbcursor = plt.axes([0.05, 0.779, 0.05, 0.05])
        axbbox = plt.axes([0.05, 0.72, 0.05, 0.05])
        axbtrash = plt.axes([0.05, 0.661, 0.05, 0.05], frameon=True, aspect="equal")
        bcursor = Button(axbcursor, "", image=cursor_img)
        bcursor.on_clicked(go2cursor)
        bbox = Button(axbbox, "", image=box_img)
        bbox.on_clicked(go2box)
        btrash = Button(axbtrash, "", image=trash_img, color="white", hovercolor="lime")
        btrash.on_clicked(go2trash)

        if len(surface["args"]) > 0:
            axbsurf = plt.axes([0.005, 0.759, 0.07, 0.07], frameon=True, aspect="equal")
            bsurf = Button(axbsurf, "", image=surface_img)
            bsurf.on_clicked(go2surface)

        plt.show(block=True)
        
    # ************************
    # SHOW SPECTRUM ALONG PATH
    # ************************
    def _plot_spectrum_path(self, fig, ax, xa, ya, chan, color_list=[], extent=None, plot_color=None, compare_cubes=[], **kwargs_curve):

        if xa is None: return 0

        if extent is None:
            j = xa.astype(int)
            i = ya.astype(int)
        else:
            nz, ny, nx = np.shape(self.data)
            dx = extent[1] - extent[0]
            dy = extent[3] - extent[2]
            j = (nx * (xa - extent[0]) / dx).astype(int)
            i = (ny * (ya - extent[2]) / dy).astype(int)

        pix_ids = np.arange(len(i))
        path_val = self.data[chan, i, j]

        if plot_color is None:
            plot_path = ax[1].step(pix_ids, path_val, where="mid", lw=2, **kwargs_curve)
            plot_color = plot_path[0].get_color()
            color_list.append(plot_color)
        else:
            plot_path = ax[1].step(pix_ids, path_val, where="mid", lw=2, color=plot_color, **kwargs_curve)
        path_on_cube = ax[0].plot(xa, ya, color=plot_color, lw=2, **kwargs_curve)
            
        ncubes = len(compare_cubes)
        if ncubes > 0:
            alpha = 0.2
            dalpha = -alpha / ncubes
            for cube in compare_cubes:
                ax[1].fill_between(
                    pix_ids,
                    cube.data[chan, i, j],
                    color=plot_color,
                    step="mid",
                    alpha=alpha,
                )
                alpha += dalpha
        else:
            ax[1].fill_between(
                pix_ids, path_val, color=plot_color, step="mid", alpha=0.1
            )

        fig.canvas.draw()
        fig.canvas.flush_events()
        
    def _curve(
        self,
        fig,
        ax,
        xa_list=[],
        ya_list=[],
        color_list=[],
        extent=None,
        click=True,
        compare_cubes=[],
        **kwargs
    ):
        kwargs_curve = dict(linewidth=2.5) 
        kwargs_curve.update(kwargs)

        xa, ya = None, None
        xm = [None]
        ym = [None]

        def mouse_move(event):
            xm[0] = event.xdata
            ym[0] = event.ydata

        def toggle_selector(event):
            toggle_selector.RS.set_active(True)

        rectprops = dict(facecolor="0.7", edgecolor="k", alpha=0.3, fill=True)
        lineprops = dict(color="white", linestyle="-", linewidth=3, alpha=0.8)
            
        def onselect(eclick, erelease):
            print ('channel_onselect:', self._chan_path)
            if eclick.inaxes is ax[0]:
                #Must correct if click and realease are not right by comparing with current pos of mouse.
                if xm[0] < erelease.xdata: 
                    eclick.xdata, erelease.xdata=erelease.xdata, eclick.xdata
                if ym[0] < erelease.ydata:
                    eclick.ydata, erelease.ydata=erelease.ydata, eclick.ydata
                x0, y0 = eclick.xdata, eclick.ydata
                x1, y1 = erelease.xdata, erelease.ydata
                xa = np.linspace(x0, x1, 100)
                ya = np.linspace(y0, y1, 100)
                print("startposition: (%.1f, %.1f)" % (x0, y0))
                print("endposition  : (%.1f, %.1f)" % (x1, y1))
                print("used button  : ", eclick.button)
                xa_list.append(xa)
                ya_list.append(ya)
                self._plot_spectrum_path(fig, ax, xa, ya, self._chan_path, color_list=color_list, extent=extent, compare_cubes=compare_cubes, **kwargs)
            
        if click:    
            toggle_selector.RS = RectangleSelector(
                ax[0], onselect, drawtype="box", rectprops=rectprops, lineprops=lineprops
            )

        cid = fig.canvas.mpl_connect("key_press_event", toggle_selector)
        fig.canvas.mpl_connect('motion_notify_event', mouse_move)        
        return cid

    def show_path(
        self,
        extent=None,
        chan_init=20,
        cube_init=0,
        compare_cubes=[],
        cursor_grid=True,
        int_unit=r"Intensity [mJy beam$^{-1}$]",
        pos_unit="au",
        vel_unit=r"km s$^{-1}$",
        show_beam=False,
        **kwargs
    ):

        self._check_cubes_shape(compare_cubes)
        v0, v1 = self.vchannels[0], self.vchannels[-1]
        dv = v1 - v0
        fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
        plt.subplots_adjust(wspace=0.25)
        ncubes = len(compare_cubes)
        self._chan_path = chan_init
        
        y0, y1 = ax[1].get_position().y0, ax[1].get_position().y1
        axcbar = plt.axes([0.47, y0, 0.03, y1 - y0])
        max_data = np.max(self.data)
        vmin, vmax = -max_data / 30, max_data
        axcbar.tick_params(direction="out")        
        ax[0].set_xlabel(pos_unit)
        ax[0].set_ylabel(pos_unit)

        ax[1].set_xlabel("Cell id along path")
        ax[1].tick_params(direction="in", right=True, labelright=False, labelleft=False)
        ax[1].set_ylabel(int_unit, labelpad=15)
        ax[1].yaxis.set_label_position("right")
        ax[1].set_ylim(vmin, vmax)
        ax[1].grid(lw=1.5, ls=":")

        cmap = plt.get_cmap("brg")
        cmap.set_bad(color=(0.9, 0.9, 0.9))

        if show_beam and self.beam_kernel:
            self._plot_beam(ax[0])
            
        if cube_init == 0:
            img_data = self.data[chan_init]
        else:
            img_data = compare_cubes[cube_init-1].data[chan_init]
        
        img = ax[0].imshow(
            img_data,
            cmap=cmap,
            extent=extent,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        cbar = plt.colorbar(img, cax=axcbar)
        text_chan = ax[1].text(
            0.15,
            1.04,  # Converting xdata coords to Axes coords
            r"v$_{\rmchan}$=%4.1f %s" % (self.vchannels[chan_init], vel_unit),
            ha="center",
            color="black",
            transform=ax[1].transAxes,
        )

        if cursor_grid:
            cg = Cursor(ax[0], useblit=True, color="lime", linewidth=1.5)
        box_img = plt.imread(path_icons + "button_box.png")
        cursor_img = plt.imread(path_icons + "button_cursor.jpeg")

        xa_list, ya_list, color_list = [], [], []
        def get_interactive(func, click=True):
            cid = func(
                fig,
                ax,
                xa_list = xa_list,
                ya_list = ya_list,
                color_list = color_list,
                extent=extent,
                click = click,
                compare_cubes=compare_cubes,
                **kwargs
            )
            return cid
        
        interactive_obj = [get_interactive(self.interactive_path)]

        # ***************
        # SLIDERS
        # ***************
        def update_chan(val):
            chan = int(val)
            vchan = self.vchannels[chan]
            if ncubes>0:
                ci = int(slider_cubes.val)
                if ci == 0:
                    img.set_data(self.data[chan])
                else:
                    img.set_data(compare_cubes[ci-1].data[chan])
            else:
                img.set_data(self.data[chan])
            self._chan_path = chan
            for line in ax[1].get_lines(): 
                line.remove()
            for i in range(len(xa_list)): #Needs to be done more than once for some (memory) reason
                for mcoll in ax[1].collections:
                    mcoll.remove()
            text_chan.set_text(r"v$_{\rmchan}$=%4.1f %s" % (vchan, vel_unit))            
            for i in range(len(xa_list)):
                if xa_list[i] is not None:
                    self._plot_spectrum_path(fig, ax, xa_list[i], ya_list[i], chan, extent=extent, plot_color=color_list[i], compare_cubes=compare_cubes, **kwargs)                    
            fig.canvas.draw_idle()

        def update_cubes(val):
            ci = int(val)
            chan = int(slider_chan.val)
            vchan = self.vchannels[chan]
            if ci == 0:
                img.set_data(self.data[chan])
            else:
                img.set_data(compare_cubes[ci-1].data[chan])
            fig.canvas.draw_idle()
            
        if ncubes > 0:
            axcubes = plt.axes([0.2, 0.90, 0.24, 0.025], facecolor="0.7")
            axchan = plt.axes([0.2, 0.95, 0.24, 0.025], facecolor="0.7")
            slider_cubes = Slider(
                axcubes,
                "Cube id",
                0,
                ncubes,
                valstep=1,
                valinit=cube_init,
                valfmt="%1d",
                color="dodgerblue",
            )
            slider_chan = Slider(
                axchan,
                "Channel",
                0,
                self.nchan - 1,
                valstep=1,
                valinit=chan_init,
                valfmt="%2d",
                color="dodgerblue",
            )
            slider_cubes.on_changed(update_cubes)
            slider_chan.on_changed(update_chan)
        else:
            axchan = plt.axes([0.2, 0.9, 0.24, 0.05], facecolor="0.7")
            slider_chan = Slider(
                axchan,
                "Channel",
                0,
                self.nchan - 1,
                valstep=1,
                valinit=chan_init,
                valfmt="%2d",
                color="dodgerblue",
            )
            slider_chan.on_changed(update_chan)

        # *************
        # BUTTONS
        # *************            
        def go2trash(event):
            print("Cleaning interactive figure...")
            plt.close()
            chan = int(slider_chan.val)
            if ncubes>0: ci = int(slider_cubes.val)
            else: ci=0
            self.show_path(
                extent=extent,
                chan_init=chan,
                cube_init=ci,
                compare_cubes=compare_cubes,
                cursor_grid=cursor_grid,
                int_unit=int_unit,
                pos_unit=pos_unit,
                vel_unit=vel_unit,
                show_beam=show_beam,
                **kwargs
            )
            
        trash_img = plt.imread(path_icons + "button_trash.jpg")
        axbtrash = plt.axes([0.05, 0.661, 0.05, 0.05], frameon=True, aspect="equal")
        btrash = Button(axbtrash, "", image=trash_img, color="white", hovercolor="lime")
        btrash.on_clicked(go2trash)
                        
        plt.show(block=True)

    # *************
    # MAKE GIF
    # *************        
    def make_gif(
        self,
        folder="./gif/",
        extent=None,
        velocity2d=None,
        unit=r"Brightness Temperature [K]",
        gif_command="convert -delay 10 *int2d* cube_channels.gif",
    ):
        cwd = os.getcwd()
        if folder[-1] != "/":
            folder += "/"
        os.system("mkdir %s" % folder)
        max_data = np.max(self.data)

        clear_list, coll_list = [], []
        fig, ax = plt.subplots()
        contour_color = "red"
        cmap = plt.get_cmap("binary")
        cmap.set_bad(color=(0.9, 0.9, 0.9))
        ax.plot(
            [None],
            [None],
            color=contour_color,
            linestyle="--",
            linewidth=2,
            label="Upper surface",
        )
        ax.plot(
            [None],
            [None],
            color=contour_color,
            linestyle=":",
            linewidth=2,
            label="Lower surface",
        )
        ax.set_xlabel("au")
        ax.set_ylabel("au")

        for i in range(self.nchan):
            vchan = self.vchannels[i]
            int2d = ax.imshow(
                self.data[i], cmap=cmap, extent=extent, origin="lower", vmax=max_data
            )
            cbar = plt.colorbar(int2d)
            cbar.set_label(unit)
            if velocity2d is not None:
                vel_near = ax.contour(
                    velocity2d["upper"],
                    levels=[vchan],
                    colors=contour_color,
                    linestyles="--",
                    linewidths=1.3,
                    extent=extent,
                )
                vel_far = ax.contour(
                    velocity2d["lower"],
                    levels=[vchan],
                    colors=contour_color,
                    linestyles=":",
                    linewidths=1.3,
                    extent=extent,
                )
                coll_list = [vel_near, vel_far]
            text_chan = ax.text(
                0.7, 1.02, "%4.1f km/s" % vchan, color="black", transform=ax.transAxes
            )
            ax.legend(loc="upper left")
            plt.savefig(folder + "int2d_chan%04d" % i)

            clear_list = [cbar, int2d, text_chan]
            for obj in clear_list:
                obj.remove()
            for obj in coll_list:
                for coll in obj.collections:
                    coll.remove()
        plt.close()
        os.chdir(folder)
        print("Making movie...")
        os.system(gif_command)
        os.chdir(cwd)
