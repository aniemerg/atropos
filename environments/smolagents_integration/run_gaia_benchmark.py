#!/usr/bin/env python3
"""
Script to run GAIA benchmark with Atropos and SmolaGents integration.

This script demonstrates how to set up and run the GAIA benchmark environment
using the AtroposServerModel to bridge Atropos and SmolaGents.
"""

import argparse
import asyncio
import logging
import os

from atroposlib.envs.server_handling.openai_server import OpenaiConfig
from atroposlib.envs.server_handling.server_manager import ServerManager
from environments.smolagents_integration.gaia_benchmark_env import (
    GAIABenchmarkConfig,
    GAIABenchmarkEnv,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GAIA benchmark with Atropos-SmolaGents integration"
    )

    # Environment configuration
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/gaia",
        help="Path to the GAIA benchmark data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use (validation, test)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=12,
        help="Maximum number of steps for the agent",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--use-chat-completion",
        action="store_true",
        help="Use chat completion API instead of completion API",
    )

    # Server configuration
    parser.add_argument(
        "--api-key",
        type=str,
        default="x",
        help="API key for OpenAI API. Use 'x' for local servers.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of the API endpoint",
    )
    parser.add_argument(
        "--model-name", type=str, default="gpt-3.5-turbo", help="Model name to use"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout for server requests in seconds",
    )

    # Wandb configuration
    parser.add_argument(
        "--use-wandb", action="store_true", help="Whether to use wandb for logging"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default="gaia-benchmark",
        help="Name to use for wandb run",
    )

    return parser.parse_args()


async def main():
    """Main function to set up and run the GAIA benchmark."""
    args = parse_args()

    # Create server configuration
    server_config = OpenaiConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        timeout=args.timeout,
    )

    # Create environment configuration
    env_config = GAIABenchmarkConfig(
        dataset_path=args.dataset_path,
        split=args.split,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        use_chat_completion=args.use_chat_completion,
        use_wandb=args.use_wandb,
        wandb_name=args.wandb_name,
    )

    # Create and run the environment
    env = GAIABenchmarkEnv(
        config=env_config,
        server_configs=server_config,
        slurm=False,
        testing=False,
    )

    logger.info("Starting GAIA benchmark environment")

    # Run the environment's manager
    await env.env_manager()


if __name__ == "__main__":
    asyncio.run(main())
