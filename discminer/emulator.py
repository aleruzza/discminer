import numpy as np

import sys

sys.path.append("../unet_emu")

from .create_model import create_nnmodel
import torch
from scipy.interpolate import griddata
from . import units as u
import importlib.util
import sys
from pathlib import Path


def hypot_func(x, y):
    return np.sqrt(x**2 + y**2)


def load_params(params_path):
    """Dynamically loads the params.py file and extracts the 'params' dictionary."""
    params_path = Path(params_path).resolve()
    module_name = "params_module"

    if not params_path.exists():
        raise FileNotFoundError(f"Params file not found: {params_path}")

    spec = importlib.util.spec_from_file_location(module_name, params_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "params"):
        raise AttributeError("params.py does not contain a 'params' dictionary")

    return module.params


def generate_ict_128x128_disc_tri(slopes, dimension):
    x = np.linspace(-3, 3, dimension)
    y = np.linspace(-3, 3, dimension)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    vaz_ict = np.float32(r ** (-0.5) * ((r < 3) & (r > 0.4)))
    vaz_ict = np.expand_dims(
        np.repeat(np.expand_dims(vaz_ict, 0), len(slopes), axis=0), 1
    )
    vr_ict = np.zeros(vaz_ict.shape)
    dens_ict = generate_ict_128x128_disc(slopes, dimension=dimension, nonorm=True)
    ict = np.concatenate([dens_ict, vaz_ict, vr_ict], axis=1)
    return np.float32(ict)


def generate_ict_128x128_disc_tri_slopes(slopes, dimension):
    dens_ict = generate_ict_128x128_disc(slopes, dimension=dimension, nonorm=True)
    ict = np.concatenate([dens_ict, dens_ict.copy(), dens_ict.copy()], axis=1)
    return np.float32(ict)


def generate_ict_128x128_disc(slopes, dimension, nonorm=False):
    # generating initial conditions
    x = np.linspace(-3, 3, dimension)
    y = np.linspace(-3, 3, dimension)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    ict = np.float32(r ** (-slopes.reshape(-1, 1, 1)) * ((r < 3) & (r > 0.4)))
    if not nonorm:
        ict = np.float32(ict)
    ict = np.expand_dims(ict, axis=1)
    return ict
xy = np.linspace(-3,3,256)
xx, yy = np.meshgrid(xy, xy)

def norm_labels(labels):
    # ['PlanetMass', 'AspectRatio', 'Alpha',  'FlaringIndex']
    max = np.array([1e-2, 0.1, 0.01, 0.35])
    min = np.array([1e-5, 0.03, 1e-4, 0])
    for i in [0, 2]:
        labels[:, i] = np.log10(labels[:, i])
        max[i] = np.log10(max[i])
        min[i] = np.log10(min[i])
    labels = 2 * (labels - min) / (max - min) - 1
    return labels


class BaseEmulator:

    def __init__(self, model_pth="", model_para={}, device="cpu", norm_func=None):
        self.params = model_para
        self.device = device
        self.emulator = create_nnmodel(
            n_param=self.params["n_param"],
            image_size=self.params["image_size"],
            num_channels=self.params["num_channels"],
            num_res_blocks=self.params["num_res_blocks"],
            channel_mult=self.params["channel_mult"],
            mode=self.params["mode"],
            unc=self.params["unc"],
        ).to(device=torch.device(self.device))
        dataem = torch.load(model_pth, map_location=torch.device(self.device))
        self.emulator.load_state_dict(dataem)
        self.norm_func = norm_func if norm_func is not None else lambda value: value

    def emulate(self, ic, labels):
        labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
        ic = torch.tensor(ic, dtype=torch.float32, device=self.device)
        emulation = self.emulator(ic, labels)
        return self.norm_func(emulation)
    
    def __call__(self, ic, labels):
        return self.emulate(ic, labels)


class Emulator:

    def __init__(
        self,
        model_pths=[],
        model_params=[],
        labels=["dens", "vphi", "vr", "vz"],
        device="cpu",
        ict_gen=generate_ict_128x128_disc_tri,
        ict_comp_dict = {'dens':0, 'vphi':1 , 'vr': 2},
        norm_funcs = [None, None, None, None]
    ):
        self.device = device
        self.emulators = {}
        self.ict_gen = ict_gen
        self.max_image_size = 0
        self.ict_comp_dict = ict_comp_dict
        for i, key in enumerate(labels):
            params = load_params(model_params[i])
            if params['image_size'] > self.max_image_size:
                self.max_image_size = params['image_size']
            self.emulators[key] = BaseEmulator(model_pths[i], params, device=self.device, norm_func=norm_funcs[i])


    def emulate(self, alpha, h, planetMass, sigmaSlope, flaringIndex, fields=['dens', 'vphi', 'vr']):
        
        params_l = np.array([planetMass, h, alpha, flaringIndex]).reshape(1, 4)
        norm_params = norm_labels(params_l)
        result = []
        
        ic = self.ict_gen(
            slopes=np.array([sigmaSlope]), dimension=self.max_image_size
        )
        
        for i, key in enumerate(fields):
            if self.emulators[key].params['image_size'] < self.max_image_size:
                #TODO: implement interpolation to smaller size. For now just use the same size for all fields.
                raise NotImplementedError()
            result.append(self.emulators[key](ic[:, [self.ict_comp_dict[key]]], norm_params).detach()[0])
        return torch.concatenate(result, axis=0)


    def emulate_dens(self, alpha, h, planetMass, sigmaSlope, flaringIndex):
        return self.emulate(alpha, h, planetMass, sigmaSlope, flaringIndex, fields=['dens'])

    def emulate_vphi(self, alpha, h, planetMass, sigmaSlope, flaringIndex):
        return self.emulate(alpha, h, planetMass, sigmaSlope, flaringIndex, fields=['vphi'])

    def emulate_vr(self, alpha, h, planetMass, sigmaSlope, flaringIndex):
        return self.emulate(alpha, h, planetMass, sigmaSlope, flaringIndex, fields=['vr'])

    def per_b(self, t):
        shape = t.shape
        t = t.flatten()
        t[t > np.pi] = t[t > np.pi] - 2 * np.pi
        t[t < -np.pi] = t[t < -np.pi] + 2 * np.pi
        return t.reshape(*shape)

    def emulate_v2d(
        self,
        coord,
        alpha,
        h,
        planetMass,
        sigmaSlope,
        flaringIndex,
        R_p,
        phi_p,
        extrap_vfunc,
        **extrap_kwargs,
    ):

        G = 6.67384e-11
        if "Mstar" in extrap_kwargs.keys():
            Mstar = extrap_kwargs["Mstar"]
        else:
            Mstar = 1

        if "v_sign" in extrap_kwargs.keys():
            v_sign = extrap_kwargs["v_sign"]
        else:
            v_sign = 1

        v3d = (
            self.emulate(alpha, h, planetMass, sigmaSlope, flaringIndex, fields=['vphi', 'vr'])
            .detach()
            .numpy()
        )
        v3d = v3d[:, :, ::-1]

        x = np.linspace(-3, 3, self.max_image_size)
        y = np.linspace(-3, 3, self.max_image_size)
        xx, yy = np.meshgrid(x, y)

        rr = hypot_func(xx, yy)
        pp = np.arctan2(yy, xx)
        rr_dom = rr[(rr > 0.4) & (rr < 3)] * R_p
        pp_dom = self.per_b(pp[(rr > 0.4) & (rr < 3)] + phi_p)
        v3d_dom = v3d[:, (rr > 0.4) & (rr < 3)]
        x_dom = rr_dom * np.cos(pp_dom)
        y_dom = rr_dom * np.sin(pp_dom)

        if "R" not in coord.keys():
            R = hypot_func(coord["x"], coord["y"])
        else:
            R = coord["R"]

        if "phi" not in coord.keys():
            phi = np.arctan2(coord["x"], coord["y"])
        else:
            phi = coord["phi"]

        vphi_interp = (
            (
                griddata(
                    (x_dom, y_dom), v3d_dom[0].reshape(-1), (coord["x"], coord["y"])
                )
                * np.sqrt(G * Mstar * u.MSun / R_p)
            )
            * 1e-3 #this is because we use km 
            * v_sign * (-1) #the -1 is there to get the right interpolation
        )
        vr_interp = (
            griddata((x_dom, y_dom), v3d_dom[1].reshape(-1), (coord["x"], coord["y"]))
            * 1e-3 * np.sqrt(G * Mstar * u.MSun / R_p)
        )

        mask = (R > 2.9 * R_p) | (R < 0.5 * R_p)
        vphi_interp[mask] = extrap_vfunc(coord, **extrap_kwargs)[mask]
        vr_interp[mask] = 0
        v3d_interp = np.concatenate(
            [
                np.expand_dims(vphi_interp, axis=0),
                np.expand_dims(vr_interp, axis=0),
                -np.zeros((1, *vphi_interp.shape)) * v_sign,
            ],
            axis=0,
        )

        return v3d_interp
