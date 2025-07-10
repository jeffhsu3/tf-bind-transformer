import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close
from tf_bind_transformer import RMSBatchNorm1d

def test_rms_batchnorm1d_training_behavior():
    torch.manual_seed(0)
    bn = RMSBatchNorm1d(num_features=4)
    bn.train()

    x = torch.randn(10, 4, 20)  # (batch, channels, length)
    out = bn(x)

    # Expected: normalize by RMS (sqrt(mean(x^2)))
    expected_var = torch.mean(x ** 2, dim=[0, 2])  # shape: (channels,)
    inv_std = 1 / torch.sqrt(expected_var + bn.eps)
    expected = x * inv_std.view(1, -1, 1) * bn.scale.view(1, -1, 1) + bn.offset.view(1, -1, 1)

    assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_running_var_updated():
    bn = RMSBatchNorm1d(num_features=2, momentum=0.5)
    bn.train()
    
    x = torch.randn(4, 2, 10)
    prev_var = bn.running_var.clone()
    bn(x)  # forward pass
    
    # Running variance should have changed
    assert not torch.equal(prev_var, bn.running_var), "Running variance not updated"


def test_rms_batchnorm1d_eval_mode():
    bn = RMSBatchNorm1d(num_features=3)
    bn.train()
    
    x = torch.randn(8, 3, 10)
    bn(x)  # update running_var
    running_var_snapshot = bn.running_var.clone()
    
    bn.eval()
    x2 = torch.randn(8, 3, 10)
    out = bn(x2)

    # Use stored running_var, not batch var
    expected = x2 / torch.sqrt(running_var_snapshot.view(1, -1, 1) + bn.eps)
    expected = expected * bn.scale.view(1, -1, 1) + bn.offset.view(1, -1, 1)
    
    assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_gradients_flow():
    bn = RMSBatchNorm1d(num_features=5)
    bn.train()

    x = torch.randn(2, 5, 4, requires_grad=True)
    out = bn(x)
    loss = out.sum()
    loss.backward()

    assert bn.scale.grad is not None, "No gradient for scale"
    assert bn.offset.grad is not None, "No gradient for offset"
    assert x.grad is not None, "No gradient for input"


