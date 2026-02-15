#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

from omegaconf import OmegaConf

from megatron.bridge.recipes.llama.llama3 import llama31_8b_pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.training.vlm_step import forward_step
from megatron.bridge.utils.common_utils import get_rank_safe


logger: logging.Logger = logging.getLogger(__name__)


SCRIPT_DIR: Path = Path(__file__).parent.resolve()
DEFAULT_CONFIG_FILENAME: str = "llama31_pretrain.yaml"
DEFAULT_CONFIG_FILE_PATH: Path = SCRIPT_DIR / "conf" / DEFAULT_CONFIG_FILENAME


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse known script args and return remaining as Hydra-style overrides."""
    parser = argparse.ArgumentParser(
        description="Continuous pretraining Llama 3.1 with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(DEFAULT_CONFIG_FILE_PATH),
        help="Path to the YAML OmegaConf override file. Default: conf/llama3_pretrain.yaml",
    )
    parser.add_argument(
        "--data-path",
        nargs='*', 
        default=None,
        help="Path to tokenized dataset (preloaded conversation or legacy messages format).",
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        default="meta-llama/Llama3.1-8B",
        help="hf checkpoint to load."
    )
    parser.add_argument(
        "--use-mpi",
        action="store_true",
        help="Use MPI for distributed training."
    )
    parser.add_argument(
        "--wandb-exp-name",
        default="",
        help="Wandb experiment name."
    )
    
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """
    Load the base VLM recipe config, apply YAML/CLI overrides, and start pretraining.
    """
    args, cli_overrides = parse_cli_args()

    logger.info("Megatron-Bridge Llama 3.1 8B Continous Pre-Training Script with YAML & CLI Overrides")
    logger.info("-----------------------------------------------------------------------")

    if args.use_mpi:
        os.environ["RANK"] = os.getenv("OMPI_COMM_WORLD_RANK", "0")
        os.environ["LOCAL_RANK"] = os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0")
        os.environ["WORLD_SIZE"] = os.getenv("OMPI_COMM_WORLD_SIZE", "1")
    
    cfg: ConfigContainer = llama31_8b_pretrain_config(
        hf_path = args.hf_path,
        data_paths = args.data_path,
        load_weights = True,
    )

    cfg.logger.wandb_exp_name = args.wandb_exp_name

    logger.info("Loaded base configuration")

    if get_rank_safe() == 0:
        cfg.print_yaml()

    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    if args.config_file:
        logger.debug(f"Loading YAML overrides from: {args.config_file}")
        if not os.path.exists(args.config_file):
            logger.error(f"Override YAML file not found: {args.config_file}")
            sys.exit(1)
        yaml_overrides_omega = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)

    if cli_overrides:
        logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)

    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    if get_rank_safe() == 0:
        logger.info("--- Final Merged Configuration ---")
        cfg.print_yaml()
        logger.info("----------------------------------")

    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()