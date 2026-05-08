import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
from equinox import tree_at
import sys
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

def scale_mlp_params(mlp: eqx.nn.MLP, w_scale: float = 2.0, b_scale: float = 1.0):
    """Multiply Linear weights/biases by constants."""
    def scale_leaf(x):
        # eqx.nn.Linear stores arrays; we just scale all arrays we see.
        # If you want to only scale weights and not biases, see option (2).
        if isinstance(x, jnp.ndarray):
            return x * w_scale
        return x


    # This scales *all* arrays (including biases). Often fine as a first pass.
    # tree_map doesn't seem to exist and recommended tree_at doesn't work with this call signature
    # return eqx.tree_map(scale_leaf, mlp)
    # return eqx.tree_at(scale_leaf,mlp)
    return jtu.tree_map(scale_leaf,mlp)

class CNN(eqx.Module):
    layers: list
    final_size: int

    def __init__(self, in_channels, in_size, out_features, *, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        channel_progression = [in_channels, in_channels, in_channels, in_channels]
        self.final_size = in_size * in_channels // 8
        self.layers = [
            eqx.nn.Conv1d(
                in_channels=channel_progression[0],
                out_channels=channel_progression[1],
                kernel_size=4,
                stride=2,
                padding=1,
                key=key1,
            ),
            jax.nn.relu,
            eqx.nn.Conv1d(
                in_channels=channel_progression[1],
                out_channels=channel_progression[2],
                kernel_size=4,
                stride=2,
                padding=1,
                key=key2,
            ),
            jax.nn.relu,
            eqx.nn.Conv1d(
                in_channels=channel_progression[2],
                out_channels=channel_progression[3],
                kernel_size=4,
                stride=2,
                padding=1,
                key=key3,
            ),
            jax.nn.relu,
            eqx.nn.Linear(in_features=self.final_size, out_features=out_features, key=key4),
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](x.reshape(self.final_size))
        return x


class HohlNet(eqx.Module):
    nn_input_laser_leh: CNN
    nn_mlp_leh: eqx.Module
    nn_input_laser_hohl: CNN
    nn_mlp_hohl: eqx.Module
    nn_input_laser_sbs_source: CNN
    nn_mlp_sbs_source: eqx.Module
    leh_outputs: list
    hohl_outputs: list
    sbs_source_outputs: list
    quantity_bounds: dict
    model_cfg: dict

    def __init__(self, beams, t_pts, n_design_params, seed=0):
        self.model_cfg = dict(beams=beams, t_pts=t_pts, n_design_params=n_design_params, seed=seed)
        seed = int(seed)
        cnn_out_features = 64
        depth = 3
        width = 16

        self.leh_outputs = [
            "n_over_n0",
            "Te_over_T0",
            "Ti_over_T0",
            "flow_magnitude",
            "flow_theta",
            "flow_phi",
            "Zeff",
            "A_ion",
            "L_p",
        ]
        self.quantity_bounds = {'n_over_n0':[1e-4,0.25],
                                'Te_over_T0':[0.01,8],
                                'Ti_over_T0':[0.01,8],
                                'flow_over_flow0':[0,5],
                                'flow_magnitude':[0,1e-3],
                                'flow_theta':[0,jnp.pi],
                                'flow_phi':[0,2*jnp.pi],
                                'Zeff':[1,100],
                                'A_ion':[1,300],
                                'L_p': [5,1e4/0.33],
                                'omegabeat':[1e11,1e13],
                                'thermal_noise':[1e-10,1e-1]}

        # self.quantity_bounds = {
        #     "n_over_n0": [1e-3, 2.5e-2],
        #     "Te_over_T0": [1.0, 2.0],
        #     "Ti_over_T0": [1.0 / 3, 2.0 / 3],
        #     "flow_over_flow0": [0, 1e-3],
        #     "flow_magnitude": [0, 1e-5],
        #     "flow_theta": [0, jnp.pi],
        #     "flow_phi": [0, 2 * jnp.pi],
        #     "Zeff": [3, 5],
        #     "A_ion": [10, 15],
        #     "L_p": [400.0, 800.0],
        #     "omegabeat": [1e12, 1e13],
        #     "thermal_noise": [1e-10, 1e-1],
        # }
        # inputs = embedded input pulse, design inputs, t
        # outputs = n, Te, Ti, flow_magnitude, flow_theta, flow_phi, Zeff, A_ion, Lp

        key = jax.random.PRNGKey(seed)
        (
            key_leh,
            key_hohl,
            key_source,
            key_cnn_leh,
            key_cnn_hohl,
            key_cnn_source,
            key_vmap_design,
        ) = jax.random.split(key, 7)

        self.nn_input_laser_leh = CNN(beams, t_pts, cnn_out_features, key=key_cnn_leh)
        self.nn_mlp_leh = eqx.nn.MLP(cnn_out_features + n_design_params + 1, 9, width_size=width, depth=depth, key=key_leh)

        self.nn_input_laser_hohl = CNN(beams, t_pts, cnn_out_features, key=key_cnn_hohl)
        self.nn_mlp_hohl = eqx.nn.MLP(cnn_out_features + n_design_params + 3, 7, width_size=width, depth=depth, key=key_hohl)
        self.hohl_outputs = ["n_over_n0", "Te_over_T0", "Ti_over_T0", "flow_over_flow0", "Zeff", "A_ion", "omegabeat"]
        # inputs = embedded input pulse, design inputs, subcone index, z, t
        # outputs = n, Te, Ti, flow (along beam), Zeff, A_ion, omegabeat

        self.nn_input_laser_sbs_source = CNN(beams, t_pts, cnn_out_features, key=key_cnn_source)
        self.nn_mlp_sbs_source = eqx.nn.MLP(
            cnn_out_features + n_design_params + 2, 1, width_size=width, depth=depth, key=key_source
        )
        self.sbs_source_outputs = ["thermal_noise"]
        # inputs = embedded input pulse, design inputs, subcone index, t
        # outputs = thermal_noise

        repeats = 10
        ts = jnp.linspace(0,1,t_pts)
        vmap_in_laser = jnp.tile(ts**2,reps = (repeats,beams,1))
        vmap_design = jax.random.uniform(key=key_vmap_design, shape=(repeats, n_design_params))
        vmap_z = jnp.linspace(0,1,repeats)
        vmap_t = jnp.linspace(0,1,repeats)
        vmap_beams = jnp.arange(repeats)

        beam_outs = np.stack(
            [
                jax.flatten_util.ravel_pytree(
                    self.hohl_eval(
                        vmap_in_laser[i],
                        vmap_design[i],
                        vmap_beams[i:i+1],
                        vmap_z[i:i+1],
                        vmap_t[i:i+1],
                    )
                )[0]
                for i in range(repeats)
            ],
            axis=0,
        )
        variation = np.std(beam_outs, axis=0) / np.abs(np.mean(beam_outs, axis=0))
        print(variation)

        low_scale = 1.0
        n_steps = 0
        while variation.mean() < 0.1 and n_steps < 10:
            n_steps += 1
            low_scale += 1.0
            iter_seed = seed + 1000 * n_steps
            iter_rng = np.random.default_rng(iter_seed)
            key = jax.random.PRNGKey(iter_seed)
            key_leh, key_hohl, key_source, key_cnn_leh, key_cnn_hohl, key_cnn_source, key_vmap_design = jax.random.split(key, 7)
            vmap_design = jax.random.uniform(key=key_vmap_design, shape=(repeats, n_design_params))

            self.nn_input_laser_leh = CNN(beams, t_pts, cnn_out_features, key=key_cnn_leh)
            self.nn_mlp_leh = eqx.nn.MLP(
                cnn_out_features + n_design_params + 1, 9, width_size=width, depth=depth, key=key_leh
            )
            self.nn_mlp_leh = scale_mlp_params(
                self.nn_mlp_leh, w_scale=iter_rng.uniform(low=low_scale, high=1.5 * low_scale)
            )

            self.nn_input_laser_sbs_source = CNN(beams, t_pts, cnn_out_features, key=key_cnn_source)
            self.nn_mlp_sbs_source = eqx.nn.MLP(
                cnn_out_features + n_design_params + 2, 1, width_size=width, depth=depth, key=key_source
            )
            self.nn_mlp_sbs_source = scale_mlp_params(
                self.nn_mlp_sbs_source, w_scale=iter_rng.uniform(low=low_scale, high=1.5 * low_scale)
            )

            self.nn_input_laser_hohl = CNN(beams, t_pts, cnn_out_features, key=key_cnn_hohl)
            self.nn_mlp_hohl = eqx.nn.MLP(
                cnn_out_features + n_design_params + 3, 7, width_size=width, depth=depth, key=key_hohl
            )
            self.nn_mlp_hohl = scale_mlp_params(
                self.nn_mlp_hohl, w_scale=iter_rng.uniform(low=low_scale, high=1.5 * low_scale)
            )

            beam_outs = np.stack(
                [
                    jax.flatten_util.ravel_pytree(
                        self.hohl_eval(
                            vmap_in_laser[i],
                            vmap_design[i],
                            vmap_beams[i:i+1],
                            vmap_z[i:i+1],
                            vmap_t[i:i+1],
                        )
                    )[0]
                    for i in range(repeats)
                ],
                axis=0,
            )
            variation = np.std(beam_outs, axis=0) / np.abs(np.mean(beam_outs, axis=0))
            print(variation)
    def save(self, file_path):
        """
        Save the model to a file

        Parameters
        ----------
        filename : str
            The name of the file to save the model to

        """

        with open(file_path, "wb") as f:
            model_cfg_str = json.dumps(self.model_cfg)
            f.write((model_cfg_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self)

    def leh_eval(self, input_powers, design_inputs, time):
        embedded_input_laser = self.nn_input_laser_leh(input_powers)
        x_in = jnp.concatenate([embedded_input_laser, design_inputs, time], axis=0)
        leh_plasma = self.nn_mlp_leh(x_in)
        leh_plasma_dict = {
            key: jax.nn.sigmoid(leh_plasma[i]) * (self.quantity_bounds[key][1] - self.quantity_bounds[key][0])
            + self.quantity_bounds[key][0]
            for i, key in enumerate(self.leh_outputs)
        }
        # leh_plasma_dict = {self.leh_outputs[i]:leh_plasma[i] for i in range(len(self.leh_outputs))}
        # leh_plasma_dict = {key:jax.nn.sigmoid(val)*(self.quantity_bounds[key][1]-self.quantity_bounds[key][0])
        #                    +self.quantity_bounds[key][0] for key,val in leh_plasma_dict.items()}
        return leh_plasma_dict

    def hohl_eval(self, input_powers, design_inputs, subcone_index, z, t):
        embedded_input_laser = self.nn_input_laser_hohl(input_powers)
        x_in = jnp.concatenate([embedded_input_laser, design_inputs, subcone_index, z, t], axis=0)
        hohl_plasma = self.nn_mlp_hohl(x_in)
        hohl_plasma_dict = {
            key: jax.nn.sigmoid(hohl_plasma[i]) * (self.quantity_bounds[key][1] - self.quantity_bounds[key][0])
            + self.quantity_bounds[key][0]
            for i, key in enumerate(self.hohl_outputs)
        }
        # hohl_plasma_dict = {self.hohl_outputs[i]:hohl_plasma[i] for i in range(len(self.hohl_outputs))}
        return hohl_plasma_dict

    def sbs_source_eval(self, input_powers, design_inputs, subcone_index, t):
        embedded_input_laser = self.nn_input_laser_sbs_source(input_powers)
        x_in = jnp.concatenate([embedded_input_laser, design_inputs, subcone_index, t], axis=0)
        sbs_source = self.nn_mlp_sbs_source(x_in)
        sbs_source_dict = {
            key: jax.nn.sigmoid(sbs_source[i]) * (self.quantity_bounds[key][1] - self.quantity_bounds[key][0])
            + self.quantity_bounds[key][0]
            for i, key in enumerate(self.sbs_source_outputs)
        }
        # sbs_source_dict = {self.sbs_source_outputs[i]:sbs_source[i] for i in range(len(self.sbs_source_outputs))}
        return sbs_source_dict

    def get_partition_spec(self):
        filter_spec = jtu.tree_map(lambda _: False, self)
        nn_input_laser_leh_filter_spec = jtu.tree_map(lambda _: False, self.nn_input_laser_leh)
        for i, layer in enumerate(self.nn_input_laser_leh.layers):
            if hasattr(layer, "weight"):
                nn_input_laser_leh_filter_spec = tree_at(
                    lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                    nn_input_laser_leh_filter_spec,
                    replace=(True, True),
                )

        nn_mlp_leh_filter_spec = jtu.tree_map(lambda _: False, self.nn_mlp_leh)
        for i, layer in enumerate(self.nn_mlp_leh.layers):
            if hasattr(layer, "weight"):
                nn_mlp_leh_filter_spec = tree_at(
                    lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                    nn_mlp_leh_filter_spec,
                    replace=(True, True),
                )

        nn_input_laser_hohl_filter_spec = jtu.tree_map(lambda _: False, self.nn_input_laser_hohl)
        for i, layer in enumerate(self.nn_input_laser_hohl.layers):
            if hasattr(layer, "weight"):
                nn_input_laser_hohl_filter_spec = tree_at(
                    lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                    nn_input_laser_hohl_filter_spec,
                    replace=(True, True),
                )

        nn_mlp_hohl_filter_spec = jtu.tree_map(lambda _: False, self.nn_mlp_hohl)
        for i, layer in enumerate(self.nn_mlp_hohl.layers):
            if hasattr(layer, "weight"):
                nn_mlp_hohl_filter_spec = tree_at(
                    lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                    nn_mlp_hohl_filter_spec,
                    replace=(True, True),
                )

        nn_input_laser_sbs_source_filter_spec = jtu.tree_map(lambda _: False, self.nn_input_laser_sbs_source)
        for i, layer in enumerate(self.nn_input_laser_sbs_source.layers):
            if hasattr(layer, "weight"):
                nn_input_laser_sbs_source_filter_spec = tree_at(
                    lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                    nn_input_laser_sbs_source_filter_spec,
                    replace=(True, True),
                )

        nn_mlp_sbs_source_filter_spec = jtu.tree_map(lambda _: False, self.nn_mlp_sbs_source)
        for i, layer in enumerate(self.nn_mlp_sbs_source.layers):
            if hasattr(layer, "weight"):
                nn_mlp_sbs_source_filter_spec = tree_at(
                    lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                    nn_mlp_sbs_source_filter_spec,
                    replace=(True, True),
                )

        filter_spec = tree_at(lambda tree: tree.nn_input_laser_leh, filter_spec, replace=nn_input_laser_leh_filter_spec)
        filter_spec = tree_at(lambda tree: tree.nn_mlp_leh, filter_spec, replace=nn_mlp_leh_filter_spec)
        filter_spec = tree_at(
            lambda tree: tree.nn_input_laser_hohl, filter_spec, replace=nn_input_laser_hohl_filter_spec
        )
        filter_spec = tree_at(lambda tree: tree.nn_mlp_hohl, filter_spec, replace=nn_mlp_hohl_filter_spec)
        filter_spec = tree_at(
            lambda tree: tree.nn_input_laser_sbs_source, filter_spec, replace=nn_input_laser_sbs_source_filter_spec
        )
        filter_spec = tree_at(lambda tree: tree.nn_mlp_sbs_source, filter_spec, replace=nn_mlp_sbs_source_filter_spec)

        # amp_model_filter_spec = jtu.tree_map(lambda _: False, self.amp_model)
        # for i in range(len(self.amp_model.layers)):
        #     amp_model_filter_spec = tree_at(
        #         lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
        #         amp_model_filter_spec,
        #         replace=(True, True),
        #     )

        # phase_model_filter_spec = jtu.tree_map(lambda _: False, self.phase_model)
        # for i in range(len(self.phase_model.layers)):
        #     phase_model_filter_spec = tree_at(
        #         lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
        #         phase_model_filter_spec,
        #         replace=(True, True),
        #     )

        # filter_spec = tree_at(lambda tree: tree.phase_model, filter_spec, replace=phase_model_filter_spec)
        # filter_spec = tree_at(lambda tree: tree.amp_model, filter_spec, replace=amp_model_filter_spec)

        return filter_spec
