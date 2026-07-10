import numpy as np
import torch
import tempfile
from pathlib import Path

from flashace.calculator import TransformersACECalculator
from flashace.model import TransformersACEV3
from transformers_ace.deploy import LAMMPSEnergyModel


def _model():
    torch.manual_seed(23)
    return TransformersACEV3(
        r_max=4.5,
        l_max=2,
        num_radial=4,
        hidden_dim=8,
        num_layers=1,
        correlation_order=4,
        correlation_channels=4,
        attention_num_heads=2,
    ).eval()


def _data(positions):
    n_atoms = positions.shape[0]
    senders, receivers = [], []
    for receiver in range(n_atoms):
        for sender in range(n_atoms):
            if sender != receiver:
                senders.append(sender)
                receivers.append(receiver)
    return {
        "z": torch.tensor([55, 82, 53, 53][:n_atoms], dtype=torch.long),
        "pos": positions,
        "cell": torch.eye(3) * 9.0,
        "edge_index": torch.tensor([senders, receivers], dtype=torch.long),
        "edge_shift": torch.zeros((len(senders), 3)),
        "volume": torch.tensor(729.0),
    }


def test_v3_tokens_include_fixed_ace_moments_for_each_center():
    model = _model()
    data = _data(
        torch.tensor(
            [[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.5, 1.3, 0.8], [0.8, 0.7, 1.7]]
        )
    )
    edge_vec = data["pos"][data["edge_index"][0]] - data["pos"][data["edge_index"][1]]
    edge_len = torch.linalg.norm(edge_vec, dim=1)
    _, tokens, receiver, token_length, token_cutoff, token_kind = model.ace(
        model.emb(data["z"]), data["edge_index"], edge_vec, edge_len
    )
    n_atoms = len(data["z"])
    n_edges = data["edge_index"].shape[1]
    assert tokens.shape[0] == n_edges + n_atoms * model.ace.num_moment_tokens
    assert receiver.shape[0] == tokens.shape[0]
    assert token_kind[:n_edges].eq(0).all()
    assert set(token_kind[n_edges:].tolist()) == set(range(1, model.ace.num_moment_tokens + 1))
    assert token_length[n_edges:].eq(0).all()
    assert token_cutoff[n_edges:].eq(1).all()


def test_v3_attention_never_reads_an_updated_neighbor_state():
    model = _model()
    layer = model.layers[0]
    node_features = torch.randn(3, model.attention_irreps.dim)
    tokens = torch.randn(2, model.ace.irreps_correlation.dim)
    receiver = torch.tensor([0, 0], dtype=torch.long)
    length = torch.tensor([1.0, 1.3])
    cutoff = model.ace.cutoff(length)
    kind = torch.zeros(2, dtype=torch.long)

    first = layer(node_features, tokens, receiver, length, cutoff, kind)
    changed = node_features.clone()
    changed[1] += 100.0
    changed[2] -= 100.0
    second = layer(changed, tokens, receiver, length, cutoff, kind)
    torch.testing.assert_close(first[0], second[0])


def test_v3_energy_and_forces_are_rotation_equivariant():
    model = _model()
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.5, 1.3, 0.8], [0.8, 0.7, 1.7]]
    )
    energy, forces, _, _ = model(_data(positions), training=False, compute_stress=False)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated_energy, rotated_forces, _, _ = model(
        _data(positions @ rotation.T), training=False, compute_stress=False
    )
    np.testing.assert_allclose(
        float(rotated_energy.detach()), float(energy.detach()), rtol=3e-5, atol=3e-5
    )
    np.testing.assert_allclose(
        rotated_forces.detach().numpy(),
        (forces @ rotation.T).detach().numpy(),
        rtol=3e-4,
        atol=3e-4,
    )


def test_v3_stress_matches_symmetric_finite_difference():
    model = _model()
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.5, 1.3, 0.8], [0.8, 0.7, 1.7]]
    )
    data = _data(positions)
    _, _, stress, _ = model(data, training=False, compute_stress=True)
    step = 1.0e-3
    deformation = torch.eye(3)
    deformation[0, 1] = deformation[1, 0] = step
    inverse = torch.eye(3)
    inverse[0, 1] = inverse[1, 0] = -step
    e_plus = model(_data(positions @ deformation), training=False, compute_stress=False)[0]
    e_minus = model(_data(positions @ inverse), training=False, compute_stress=False)[0]
    derivative = float(((e_plus - e_minus) / (2.0 * step * data["volume"])).detach())
    np.testing.assert_allclose(derivative, 2.0 * float(stress[0, 1].detach()), rtol=3e-2, atol=4e-4)


def test_v3_lammps_wrapper_supports_position_and_strain_gradients():
    model = _model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    deploy = LAMMPSEnergyModel(model)
    data = _data(
        torch.tensor(
            [[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.5, 1.3, 0.8], [0.8, 0.7, 1.7]]
        )
    )
    pos = data["pos"].clone().requires_grad_(True)
    strain = torch.zeros(6, requires_grad=True)
    energy = deploy(
        data["z"],
        pos,
        data["cell"],
        data["edge_index"],
        data["edge_shift"],
        strain,
        torch.ones(len(data["z"])),
    )
    grad_pos, grad_strain = torch.autograd.grad(energy, (pos, strain))
    assert torch.isfinite(energy)
    assert torch.isfinite(grad_pos).all()
    assert torch.isfinite(grad_strain).all()

    traced = torch.jit.trace(
        deploy,
        (
            data["z"],
            data["pos"],
            data["cell"],
            data["edge_index"],
            data["edge_shift"],
            torch.zeros(6),
            torch.ones(len(data["z"])),
        ),
        check_trace=False,
    )
    traced_energy = traced(
        data["z"],
        data["pos"],
        data["cell"],
        data["edge_index"],
        data["edge_shift"],
        torch.zeros(6),
        torch.ones(len(data["z"])),
    )
    torch.testing.assert_close(traced_energy, energy.detach())


def test_v3_checkpoint_loads_through_the_ase_calculator():
    model = _model()
    checkpoint = {
        "config": {
            "architecture_version": 3,
            "r_max": 4.5,
            "l_max": 2,
            "num_radial": 4,
            "hidden_dim": 8,
            "num_layers": 1,
            "correlation_order": 4,
            "correlation_channels": 4,
            "radial_mlp_hidden": 32,
            "attention_num_heads": 2,
        },
        "model_state_dict": model.state_dict(),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace_v3.pt"
        torch.save(checkpoint, path)
        calculator = TransformersACECalculator(str(path), device="cpu")
        assert isinstance(calculator.model, TransformersACEV3)
