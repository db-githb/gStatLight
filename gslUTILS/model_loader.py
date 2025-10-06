# In gslUTILS/model_loader.py
from pathlib import Path
import os
import urllib.request
from rich.progress import Progress, DownloadColumn, TransferSpeedColumn

# Official SAM2.1 model URLs
MODEL_URLS = {
    "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}

def get_cache_dir():
    """Get the cache directory for gStatLight models."""
    cache_dir = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    model_dir = cache_dir / 'gstatlight' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

def get_model_path():
    """Get path to model, using cache directory."""
    model_path = get_cache_dir() / "sam2.1_hiera_large.pt"
    
    if not model_path.exists():
        download_model()
    
    return model_path

def download_model(force=False):
    """Download a specific model to cache."""
    model_name = "sam2.1_hiera_large.pt"
    model_dir = get_cache_dir()
    model_path = model_dir / model_name
    
    if model_path.exists() and not force:
        print(f"✓ Model already exists at {model_path}")
        return model_path
    
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    print(f"Downloading {model_name} from Meta...")
    print(f"URL: {url}")
    print(f"Destination: {model_path}")
    
    try:
        with Progress(
            *Progress.get_default_columns(),
            DownloadColumn(),
            TransferSpeedColumn(),
        ) as progress:
            
            def reporthook(block_num, block_size, total_size):
                if not hasattr(reporthook, 'task'):
                    reporthook.task = progress.add_task(
                        f"[cyan]Downloading {model_name}...", 
                        total=total_size
                    )
                progress.update(reporthook.task, completed=block_num * block_size)
            
            urllib.request.urlretrieve(url, model_path, reporthook=reporthook)
        
        print(f"✓ Successfully downloaded to {model_path}")
        return model_path
        
    except Exception as e:
        if model_path.exists():
            model_path.unlink()  # Clean up partial download
        raise RuntimeError(f"Failed to download model: {e}")