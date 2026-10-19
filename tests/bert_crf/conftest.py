import pytest
import torch


@pytest.fixture(autouse=True)
def synthetic_seed():
    torch.manual_seed(1729)
    torch.set_num_threads(1)


def pytest_addoption(parser):
    parser.addoption('--pretrained-dir', default=None)
    parser.addoption('--smoke-output', default=None)
    parser.addoption('--checkpoint-dir', default=None)
