# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations
import os
import sys
from typing import Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataclasses import dataclass
from pathlib import Path
from gslPY.main.utils_main import (
    write_ply,
    load_config,
)

from gslPY.main.utils_cull import (
    statcull,
    statcull_radius_std,
    statcull_mahalanobis,
    modify_model,
)

from gslPY.main.utils_ground import (
    find_ground_plane,
    rotate_ground_gaussians
)

from gslPY.main.utils_rich import CONSOLE, TABLE
from rich.panel import Panel

import time
from contextlib import contextmanager

def _pct(n, d):  # safe percent string
    return f"{(n/d):.1%}" if d else "n/a"

@contextmanager
def step(console, title, emoji=":arrow_forward:"):
    t0 = time.perf_counter()
    console.log(f"{emoji} [bold]{title}[/bold]")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        console.log(f":white_check_mark: Done [dim]({dt:.2f}s)[/dim]")

def log_totals(console, before, after, noun="splats"):
    culled = before - after
    console.log(
        f"• Removed {culled}/{before} {noun} "
        f"([bold]{_pct(culled, before)}[/bold]) → total: [bold]{after}[/bold]"
    )


@dataclass
class BaseGSL:
    """Base class for rendering."""
    model_path: Path
    """Path to model file."""

@dataclass
class gStatLight(BaseGSL):
    """Main driver class"""

    # main/driver function
    def run_statlight(self):
        count = 1
        # load model
        config, pipeline = load_config(
            self.model_path,
            test_mode="inference",
        )
        config.datamanager.dataparser.train_split_fraction = 1.0 # put all images in train split
        config.datamanager.dataparser.downscale_factor = 1
        
        # Phase 1 — statistical cull
        starting_total = pipeline.model.means.shape[0]
        with step(CONSOLE, f"Phase {count} — Statistical cull", emoji=":broom:"):
            cull_mask = statcull_mahalanobis(pipeline, threshold=0.3)
            keep = ~cull_mask
            pipeline.model = modify_model(pipeline.model, keep)
            statcull_total = pipeline.model.means.shape[0]
            log_totals(CONSOLE, starting_total, statcull_total)
            count+=1
            
        # Phase 2 — ground plane via RANSAC
        with step(CONSOLE, f"Phase {count} — Find ground plane (RANSAC)", emoji=":triangular_ruler:"):
            keep, is_ground, norm, offset = find_ground_plane(pipeline.model)
            before = pipeline.model.means.shape[0]
            #pipeline.model = rotate_ground_gaussians(is_ground, norm, pipeline.model)
            pipeline.model = modify_model(pipeline.model, keep)
            after = pipeline.model.means.shape[0]
            nvals = norm.detach().cpu().tolist()
            CONSOLE.log(f"• Plane normal: {tuple(round(float(x), 3) for x in nvals)}  offset: {float(offset):.3f}")
            log_totals(CONSOLE, before, after)
            count+=1

        # Phase 3 — write PLY
        with step(CONSOLE, f"Phase {count} — Write PLY", emoji=":floppy_disk:"):
            filename = write_ply(self.model_path, pipeline.model)
            CONSOLE.print(f"• Wrote .ply to: {filename}")
            count+=1

