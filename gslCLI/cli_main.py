import os, torch
caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(f"{m}.{n}" for m, n in sorted(caps))

import argparse
from pathlib import Path
from gslPY.main.gsl_main import gStatLight

def main():
    parser = argparse.ArgumentParser(prog="gsl", description="gsl 3DGS processing tool")
    sub = parser.add_subparsers(dest="command", required=True)
    
    p_gsl = sub.add_parser("statlight", help="Spotlight Car from 3DGS model using binary masks")
    p_gsl.add_argument("--load-config", "-l", required=True,
              help="path to 3DGS model's yaml configuration file")

    args = parser.parse_args()

    if args.command == "statlight":
        dc = gStatLight(Path(args.load_config))
        dc.run_statlight()

with torch.inference_mode():
    if __name__ == "__main__":
        main()