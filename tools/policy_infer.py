"""
Policy inference server — entry point.

Usage:
    python3 tools/policy_infer.py --policy act    --model-path /path/to/ckpt
    python3 tools/policy_infer.py --policy act_v1 --model-path /path/to/ckpt [--device cpu]
"""
import os
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

from tools.policies.runner import run_inference_server

if __name__ == "__main__":
    run_inference_server()
