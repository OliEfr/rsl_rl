#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of runners for environment-agent interaction."""

from .on_policy_runner import OnPolicyRunner
from .amp_on_policy_runner import AMPOnPolicyRunner

__all__ = ["OnPolicyRunner", "AMPOnPolicyRunner"]
