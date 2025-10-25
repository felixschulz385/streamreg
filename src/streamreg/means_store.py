"""
Dataset-local means storage using LMDB and YAML manifest.

Stores precomputed group means under dataset_root/.streamreg/means/:
- LMDB database for key-value storage (efficient read transactions)
- YAML manifest for metadata (fingerprints, write status)
"""

import lmdb
import yaml
import hashlib
import numpy as np
import pandas as pd
import pickle  # Add pickle import
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import tempfile
import shutil

logger = logging.getLogger(__name__)


class MeansStore:
    """
    LMDB-backed storage for group means with YAML manifest.
    
    Structure:
        dataset_root/.streamreg/means/
            data.mdb          # LMDB database
            lock.mdb          # LMDB lock file
            manifest.yaml     # Metadata (fingerprints, keys)
            .manifest.yaml.tmp  # Atomic write staging
    """
    
    def __init__(self, dataset_root: Path):
        """
        Initialize means store for a dataset.
        
        Parameters:
        -----------
        dataset_root : Path
            Root directory of the dataset (parquet file dir or dataframe cache location)
        """
        self.dataset_root = Path(dataset_root)
        self.store_dir = self.dataset_root / '.streamreg' / 'means'
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        self.lmdb_path = self.store_dir / 'data.mdb'
        self.manifest_path = self.store_dir / 'manifest.yaml'
        
        # Open LMDB environment (lazy - opened on first use)
        self._env = None
    
    def _open_env(self, readonly: bool = False):
        """Open LMDB environment if not already open."""
        if self._env is None:
            # Map size: 10GB (grows as needed on 64-bit systems)
            map_size = 10 * 1024 * 1024 * 1024
            self._env = lmdb.open(
                str(self.store_dir),
                map_size=map_size,
                max_dbs=0,
                readonly=readonly,
                lock=True,
                sync=True,
                map_async=False
            )

    def _load_manifest(self) -> Dict[str, Any]:
        """Load manifest from disk."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _save_manifest(self, manifest: Dict[str, Any]):
        """Atomically save manifest to disk."""
        tmp_path = self.store_dir / '.manifest.yaml.tmp'
        
        try:
            with open(tmp_path, 'w') as f:
                yaml.safe_dump(manifest, f, default_flow_style=False)
            
            # Atomic rename
            tmp_path.replace(self.manifest_path)
        except Exception as e:
            logger.warning(f"Failed to save manifest: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            raise
    
    def _compute_fingerprint(self, dataset_root: Path) -> str:
        """
        Compute dataset fingerprint for invalidation.
        
        Uses: modification times of parquet files or dataframe hash
        """
        if not dataset_root.exists():
            return "unknown"
        
        # For parquet files/directories
        if dataset_root.is_dir():
            parquet_files = sorted(dataset_root.rglob("*.parquet"))
            if parquet_files:
                # Hash of (path, mtime) tuples
                hash_input = []
                for pf in parquet_files[:100]:  # Limit to first 100 for performance
                    try:
                        stat = pf.stat()
                        hash_input.append(f"{pf.name}:{stat.st_mtime}:{stat.st_size}")
                    except:
                        pass
                
                if hash_input:
                    content = "\n".join(hash_input)
                    return hashlib.md5(content.encode()).hexdigest()
        
        # For single file
        if dataset_root.is_file():
            try:
                stat = dataset_root.stat()
                content = f"{dataset_root.name}:{stat.st_mtime}:{stat.st_size}"
                return hashlib.md5(content.encode()).hexdigest()
            except:
                pass
        
        return "unknown"
    
    def _make_key(self, variable: str, group_cols: List[str], query: Optional[str] = None) -> str:
        """
        Generate storage key with hashed grouping structure and query.
        
        The key includes:
        - Variable name
        - Hash of grouping structure (preserves combined vs separate keys)
        - Hash of query (if present)
        
        Examples:
        - "median_grp_abc123" - single group
        - "median_grp_abc123_qry_def456" - with query filter
        """
        key_parts = [variable]
        
        # Create a canonical representation of the grouping structure
        # This preserves the distinction between:
        # - ['col1', 'col2'] (two separate groups)
        # - [['col1', 'col2']] (one combined group)
        group_repr_parts = []
        for item in group_cols:
            if isinstance(item, str):
                group_repr_parts.append(item)
            elif isinstance(item, list):
                # Combined key: use parentheses to indicate grouping
                group_repr_parts.append(f"({'+'.join(sorted(item))})")
        
        # Create canonical string and hash it
        group_repr = "|".join(group_repr_parts)
        group_hash = hashlib.md5(group_repr.encode()).hexdigest()[:12]
        key_parts.append(f"grp_{group_hash}")
        
        # Add query hash if present
        if query:
            query_hash = hashlib.md5(query.encode()).hexdigest()[:12]
            key_parts.append(f"qry_{query_hash}")
        
        return "_".join(key_parts)
    
    def get(self, variable: str, group_cols: List[str], 
            query: Optional[str] = None) -> Optional[pd.Series]:
        """
        Get cached means using LMDB read transaction.
        
        Returns:
        --------
        Series with group means, or None if not cached
        """
        key = self._make_key(variable, group_cols, query)
        
        # Check manifest first (faster than LMDB lookup)
        manifest = self._load_manifest()
        if key not in manifest.get('entries', {}):
            return None
        
        # Verify fingerprint
        current_fp = self._compute_fingerprint(self.dataset_root)
        stored_fp = manifest.get('dataset_fingerprint')
        
        if stored_fp and stored_fp != current_fp:
            logger.info(f"Dataset fingerprint changed, invalidating means cache")
            return None
        
        # Read from LMDB
        try:
            self._open_env(readonly=True)
            
            with self._env.begin(write=False) as txn:
                # Keys stored as f"{key}:keys" and f"{key}:values"
                keys_data = txn.get(f"{key}:keys".encode())
                values_data = txn.get(f"{key}:values".encode())
                
                if keys_data is None or values_data is None:
                    logger.debug(f"Keys or values missing for {key}, invalidating cache entry")
                    self._invalidate_cache_entry(key, manifest)
                    return None
                
                # Deserialize: use pickle for index (object array), numpy for values
                try:
                    keys = pickle.loads(keys_data)
                    values = np.frombuffer(values_data, dtype=np.float64)
                    
                    # Validate lengths match
                    if len(keys) != len(values):
                        logger.warning(f"Length mismatch for {key}: keys={len(keys)}, values={len(values)}")
                        self._invalidate_cache_entry(key, manifest)
                        return None
                    
                except (pickle.UnpicklingError, ValueError, EOFError) as e:
                    logger.warning(f"Corrupted data for key {key}: {e}, invalidating cache entry")
                    self._invalidate_cache_entry(key, manifest)
                    return None
                
                # Create Series (no copy, uses views)
                means = pd.Series(values, index=keys, name=variable, copy=False)
                
                logger.debug(f"Means cache hit: {key}")
                return means
        
        except lmdb.Error as e:
            logger.warning(f"LMDB error loading {key}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load means from LMDB {key}: {e}")
            return None
    
    def _invalidate_cache_entry(self, key: str, manifest: Optional[Dict[str, Any]] = None):
        """Remove a corrupted cache entry from LMDB and manifest."""
        try:
            # Remove from LMDB
            if self._env is not None:
                with self._env.begin(write=True) as txn:
                    txn.delete(f"{key}:keys".encode())
                    txn.delete(f"{key}:values".encode())
            
            # Remove from manifest
            if manifest is None:
                manifest = self._load_manifest()
            
            if key in manifest.get('entries', {}):
                del manifest['entries'][key]
                self._save_manifest(manifest)
                logger.debug(f"Invalidated corrupted cache entry: {key}")
        
        except Exception as e:
            logger.debug(f"Failed to invalidate cache entry {key}: {e}")
    
    def put(self, variable: str, group_cols: List[str], means: pd.Series,
            query: Optional[str] = None):
        """
        Cache means to LMDB with atomic manifest update.
        
        Parameters:
        -----------
        variable : str
            Variable name
        group_cols : list of str
            Grouping columns
        means : Series
            Group means (index = group keys, values = means)
        query : str, optional
            Query string used for filtering
        """
        key = self._make_key(variable, group_cols, query)
        
        try:
            # Close readonly env if open, reopen in write mode
            if self._env is not None:
                self._env.close()
                self._env = None
            
            self._open_env(readonly=False)
            
            # Serialize: use pickle for index (preserves object types), numpy for values
            try:
                keys_bytes = pickle.dumps(means.index.to_numpy(), protocol=pickle.HIGHEST_PROTOCOL)
                values_bytes = means.values.astype(np.float64).tobytes()
            except Exception as e:
                logger.warning(f"Failed to serialize means for {key}: {e}")
                return
            
            # Write to LMDB (single transaction for atomicity)
            with self._env.begin(write=True) as txn:
                success_keys = txn.put(f"{key}:keys".encode(), keys_bytes, overwrite=True)
                success_values = txn.put(f"{key}:values".encode(), values_bytes, overwrite=True)
                
                if not (success_keys and success_values):
                    logger.warning(f"Failed to write to LMDB for {key}")
                    return
            
            # Verify write by reading back
            with self._env.begin(write=False) as txn:
                verify_keys = txn.get(f"{key}:keys".encode())
                verify_values = txn.get(f"{key}:values".encode())
                
                if verify_keys is None or verify_values is None:
                    logger.warning(f"Verification failed for {key}, data not readable after write")
                    return
            
            # Update manifest (atomic via tmp file)
            manifest = self._load_manifest()
            
            if 'entries' not in manifest:
                manifest['entries'] = {}
            
            manifest['entries'][key] = {
                'variable': variable,
                'group_cols': group_cols,
                'query': query,
                'n_groups': len(means),
                'mean': float(means.mean()),
                'status': 'complete'
            }
            
            manifest['dataset_fingerprint'] = self._compute_fingerprint(self.dataset_root)
            
            self._save_manifest(manifest)
            
            logger.debug(f"Cached means: {key} ({len(means):,} groups)")
        
        except lmdb.Error as e:
            logger.warning(f"LMDB error caching {key}: {e}")
        except Exception as e:
            logger.warning(f"Failed to cache means {key}: {e}")
    
    def clear(self):
        """Clear all cached means."""
        try:
            # Close environment
            if self._env is not None:
                self._env.close()
                self._env = None
            
            # Remove LMDB files
            lmdb_files = [self.store_dir / f for f in ['data.mdb', 'lock.mdb']]
            for f in lmdb_files:
                if f.exists():
                    f.unlink()
            
            # Clear manifest
            if self.manifest_path.exists():
                self.manifest_path.unlink()
            
            logger.info("Cleared means cache")
        
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
    
    def invalidate_if_stale(self) -> bool:
        """
        Check if cache is stale and invalidate if needed.
        
        Returns:
        --------
        bool: True if cache was invalidated
        """
        manifest = self._load_manifest()
        
        if not manifest:
            return False
        
        current_fp = self._compute_fingerprint(self.dataset_root)
        stored_fp = manifest.get('dataset_fingerprint')
        
        if stored_fp and stored_fp != current_fp:
            logger.info("Dataset changed, invalidating means cache")
            self.clear()
            return True
        
        return False
    
    def __del__(self):
        """Cleanup: close LMDB environment."""
        if self._env is not None:
            self._env.close()
