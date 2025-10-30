"""
Lightweight MeansStore wrapper that exposes parquet-backed means stores.

Paths used:
  <dataset_root>/.streamreg/means/<hashdataset>/<hashquery>/<hashgroupcols>.parquet

get(...) returns the uncomputed dask.DataFrame read from that parquet
(if present). No in-memory key management is performed.
"""

from pathlib import Path
from typing import Optional, List, Any
import hashlib
import logging
import time

import dask.dataframe as dd

logger = logging.getLogger(__name__)


class MeansStore:
    """
    Filesystem-backed accessor for precomputed means saved as parquet files.

    The store exposes helpers for hashing dataset paths and queries and
    returns uncomputed dask.DataFrame objects for a given grouping.
    
    Initialized with a dataset source path, it manages all path operations internally.
    """

    def __init__(self, dataset_path: Path):
        """
        Initialize MeansStore for a specific dataset.
        
        Parameters:
        -----------
        dataset_path : Path
            Full path to the dataset (file or directory). This path is used for:
            1. Hashing to create unique subdirectories per dataset
            2. Determining the root directory where .streamreg/means/ is placed
        """
        self.dataset_path = Path(dataset_path) if dataset_path is not None else None
        
        if self.dataset_path is None:
            self.base_store_dir = None
            self.dataset_hash = None
            return
        
        root_dir = self.dataset_path.parent
        
        # Create base means directory structure
        self.base_store_dir = root_dir / '.streamreg' / 'means'
        
        # Hash the full dataset path for creating unique subdirectories
        self.dataset_hash = self.hash_dataset_path(self.dataset_path)
        
        try:
            self.base_store_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.debug(f"Could not create base means dir {self.base_store_dir}: {e}")

    # ---------- Hashing utilities ----------
    @staticmethod
    def hash_dataset_path(dataset_path: Optional[Path]) -> str:
        """Deterministic hash for a dataset path (use absolute resolved path)."""
        if dataset_path is None:
            return hashlib.md5(b'').hexdigest()[:16]
        try:
            p = Path(dataset_path).resolve()
        except Exception:
            p = Path(str(dataset_path))
        return hashlib.md5(str(p).encode()).hexdigest()[:16]

    @staticmethod
    def hash_query(query: Optional[str]) -> str:
        """Deterministic hash for a query string (None -> 'noquery' hash)."""
        if query is None:
            return hashlib.md5(b'__noquery__').hexdigest()[:16]
        return hashlib.md5(query.encode()).hexdigest()[:16]

    @staticmethod
    def hash_group_cols(group_cols: List[Any]) -> str:
        """
        Deterministic hash for the grouping specification.

        This version does not canonicalize the input; it preserves the original
        structure and order when creating the hash representation. Equivalent
        inputs with different representations may yield different hashes.
        """
        # Preserve input as-is (no canonicalization)
        if group_cols is None or group_cols == '__overall__':
            group_repr = '__overall__'
        elif isinstance(group_cols, str):
            group_repr = group_cols
        else:
            # Use repr to keep a stable textual form that preserves structure/order
            try:
                group_repr = repr(group_cols)
            except Exception:
                group_repr = str(group_cols)
        return hashlib.md5(group_repr.encode()).hexdigest()[:12]

    # ---------- Path management ----------
    def get_store_path(self, query: Optional[str] = None) -> Optional[Path]:
        """
        Get the directory path for the current dataset and query.
        
        Parameters:
        -----------
        query : str, optional
            Query string for filtering
        
        Returns:
        --------
        Path or None: Directory path where means are stored
        """
        if self.base_store_dir is None or self.dataset_hash is None:
            return None
        qhash = self.hash_query(query)
        return self.base_store_dir / self.dataset_hash / qhash

    def ensure_store_path(self, query: Optional[str] = None) -> Optional[Path]:
        """
        Ensure the directory exists on disk and return the path.
        
        Parameters:
        -----------
        query : str, optional
            Query string for filtering
        
        Returns:
        --------
        Path or None: Directory path where means are stored
        """
        p = self.get_store_path(query)
        if p is None:
            return None
        p.mkdir(parents=True, exist_ok=True)
        return p

    def list_query_stores(self) -> List[Path]:
        """
        List query-store directories for the current dataset (returns full paths).
        
        Returns:
        --------
        list of Path: List of query-specific directories
        """
        res = []
        if self.base_store_dir is None or self.dataset_hash is None:
            return res
        parent = self.base_store_dir / self.dataset_hash
        if parent.exists():
            for child in parent.iterdir():
                if child.is_dir():
                    res.append(child)
        return res

    # ---------- Main API ----------
    def get(self, group_cols: List[Any], query: Optional[str] = None) -> Optional[dd.DataFrame]:
        """
        Return the uncomputed dask.DataFrame for the specified grouping and query.

        The filesystem layout is expected to contain a parquet file named:
            <hashgroupcols>.parquet
        under the directory:
            <dataset_root>/.streamreg/means/<hashdataset>/<hashquery>/

        Parameters:
        -----------
        group_cols : list
            Grouping specification
        query : str, optional
            Query string for filtering

        Returns:
        --------
        dd.DataFrame (uncomputed) if the parquet exists and can be read, None otherwise
        """
        store_dir = self.get_store_path(query)
        if store_dir is None:
            return None

        # compute group hash and expected parquet filename (hash_group_cols now normalizes input)
        ghash = self.hash_group_cols(group_cols)
        candidate_file = store_dir / f"{ghash}.parquet"

        # Accept either an explicit file or any parquet files in the directory
        try:
            if candidate_file.exists():
                return dd.read_parquet(str(candidate_file), index=group_cols if group_cols != "__overall__" else False, engine='pyarrow')
        except Exception as e:
            logger.warning(f"Failed to read parquet means at {store_dir}: {e}")
            return None

        return None

    def has(self, group_cols: List[Any], query: Optional[str] = None) -> bool:
        """
        Check whether a parquet file exists for the given grouping and query.
        
        Parameters:
        -----------
        group_cols : list
            Grouping specification
        query : str, optional
            Query string for filtering
        
        Returns:
        --------
        bool: True if parquet exists, False otherwise
        """
        store_dir = self.get_store_path(query)
        if store_dir is None:
            return False

        # hash_group_cols will canonicalize the input (handles None / strings / nested lists)
        ghash = self.hash_group_cols(group_cols)
        candidate_file = store_dir / f"{ghash}.parquet"

        try:
            if candidate_file.exists():
                return True
            # fallback: look for matching files by prefix or any parquet in dir
            if store_dir.exists():
                if any(store_dir.glob(f"{ghash}*.parquet")):
                    return True
                if any(store_dir.glob("*.parquet")):
                    return True
        except Exception as e:
            logger.debug(f"Error while checking cached means at {store_dir}: {e}")
            return False

        return False

    def refresh(self) -> None:
        """
        Refresh internal state after new means are written.
        Currently lightweight: records a timestamp and logs; kept for API compatibility.
        """
        try:
            self._last_refreshed = time.time()
        except Exception:
            self._last_refreshed = None
        logger.debug("MeansStore.refresh() called")

    def clear(self):
        """No in-memory cache to clear in this implementation (kept for API compatibility)."""
        logger.info("MeansStore.clear() called (no-op)")

    def invalidate_if_stale(self) -> bool:
        """No-op: filesystem-based reads are always read fresh."""
        return False

    def __del__(self):
        # nothing to close
        pass

    def get_temp_dir(self, prefix: str = 'streamreg_means_') -> Path:
        """
        Create and return a temporary directory for this dataset.
        
        Creates under .streamreg/tmp if possible, otherwise falls back to system temp.
        
        Parameters:
        -----------
        prefix : str
            Prefix for the temp directory name
        
        Returns:
        --------
        Path: Path to the created temp directory
        """
        if self.base_store_dir is None:
            # Fallback to system temp
            import tempfile
            return Path(tempfile.mkdtemp(prefix=prefix))
        
        tmp_base = self.base_store_dir / 'tmp'
        tmp_base.mkdir(parents=True, exist_ok=True)
        
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(tmp_base)))
        return temp_dir
