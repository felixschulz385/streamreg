"""
Demeaning module for computing and applying group means.

Handles:
- Computing group means from DataFrames (pandas groupby)
- Computing group means from Parquet files (DuckDB)
- Caching means to disk for large datasets (memory-mapped for efficiency)
- Applying demeaning transformations during estimation
"""

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import tempfile
import os
import dask.dataframe as dd  # added for lazy merges

logger = logging.getLogger(__name__)


class DemeanComputer:
    """
    Compute group means from streaming data using pandas or DuckDB.
    """
    
    def __init__(self, source_path: Optional[Path] = None):
        """
        Initialize computer.
        
        Parameters:
        -----------
        source_path : Path, optional
            Source path of the dataset for means storage. If None, no caching.
        """
        self.source_path = source_path
        self.store = None
        
        if source_path:
            from streamreg.means_store import MeansStore
            self.store = MeansStore(source_path)
            self.store.invalidate_if_stale()
    
    def compute_means(self, data, variables: List[str], group_cols: Optional[Union[str, List[Union[str, List[str]]]]] = None,
                      query: Optional[str] = None, variable_batch_size: int = 50) -> bool:
        """
        Compute group means for multiple variables.
        
        Parameters:
        -----------
        data : StreamData
            Data source
        variables : list of str
            Variables to compute means for
        group_cols : str or list of str/list or None
            Grouping specification. If None, compute overall means.
            Examples:
            - 'country': single column
            - ['country', 'year']: two separate columns
            - [['tile_ix', 'tile_iy', 'pixel_id'], 'year']: multi-level with combined key
            - None: overall means
        query : str, optional
            Query used for filtering (for cache key)
        variable_batch_size : int
            Maximum variables to compute per batch (to manage memory)
        
        Returns:
        --------
        bool: True if computation successful
        """
        logger.info(f"Computing group means for {len(variables)} variables by {group_cols}")
        
        # Build centralized grouping plan with cache status
        grouping_plan = self._build_grouping_plan(group_cols, variables, query)
        
        # Check if all groupings are cached
        variables_to_compute = set()
        for plan_item in grouping_plan:
            if plan_item['missing_vars']:
                variables_to_compute.update(plan_item['missing_vars'])
        
        if not variables_to_compute:
            logger.info("All means found in cache for all groupings")
            return True
        
        variables_to_compute = sorted(list(variables_to_compute))
        logger.info(f"Computing means for {len(variables_to_compute)} variable(s)")
        
        # Process variables in batches if many variables
        if len(variables_to_compute) > variable_batch_size:
            logger.info(f"Processing {len(variables_to_compute)} variables in batches of {variable_batch_size}")
            for i in range(0, len(variables_to_compute), variable_batch_size):
                batch = variables_to_compute[i:i+variable_batch_size]
                
                # Compute means for batch based on data type
                if data.info.source_type == 'dataframe':
                    success = self._compute_means_pandas(data, batch, grouping_plan, query)
                else:
                    success = self._compute_means_duckdb(data, batch, grouping_plan, query)
                
                if not success:
                    return False
        else:
            # Single batch: compute all variables at once
            if data.info.source_type == 'dataframe':
                success = self._compute_means_pandas(data, variables_to_compute, grouping_plan, query)
            else:
                success = self._compute_means_duckdb(data, variables_to_compute, grouping_plan, query)
            
            if not success:
                return False
        
        return True

    def _build_grouping_plan(self, group_cols: Optional[List[Union[str, List[str]]]], 
                            variables: List[str], 
                            query: Optional[str]) -> List[Dict]:
        """
        Build centralized plan for all groupings with cache status.
        
        Returns list of dicts with structure:
        {
            'type': '__overall__' | '__level__' | '__combined__',
            'cols': list of column names (empty for overall),
            'hash_spec': grouping spec for hashing,
            'cached': bool,
            'existing_vars': list of variables already in cache,
            'missing_vars': list of variables that need computation,
            'needs_merge': bool (True if we need to merge new vars with existing)
        }
        """
        plan = []
        
        # Always include overall grouping
        overall_item = {
            'type': '__overall__',
            'cols': [],
            'hash_spec': '__overall__',
            'cached': False,
            'existing_vars': [],
            'missing_vars': variables,
            'needs_merge': False
        }
        
        # Check cache status for overall
        if self.store:
            cached, existing_vars = self._check_cache_fast('__overall__', variables, query)
            overall_item['cached'] = cached
            overall_item['existing_vars'] = existing_vars
            overall_item['missing_vars'] = [v for v in variables if v not in existing_vars]
            overall_item['needs_merge'] = cached and len(existing_vars) > 0 and len(overall_item['missing_vars']) > 0
        
        plan.append(overall_item)
        
        # Add per-level and combined groupings if specified
        if group_cols is not None:            
            # Per-level groupings
            for level in group_cols:
                cols = list(level)
                level_item = {
                    'type': '__level__',
                    'cols': cols,
                    'hash_spec': level,
                    'cached': False,
                    'existing_vars': [],
                    'missing_vars': variables,
                    'needs_merge': False
                }
                
                if self.store:
                    cached, existing_vars = self._check_cache_fast(level, variables, query)
                    level_item['cached'] = cached
                    level_item['existing_vars'] = existing_vars
                    level_item['missing_vars'] = [v for v in variables if v not in existing_vars]
                    level_item['needs_merge'] = cached and len(existing_vars) > 0 and len(level_item['missing_vars']) > 0
                
                plan.append(level_item)
            
            # # Combined grouping if multi-level
            # if len(levels) > 1:
            #     all_cols = self._all_group_columns(levels)
            #     combined_item = {
            #         'type': '__combined__',
            #         'cols': all_cols,
            #         'hash_spec': group_cols,
            #         'cached': False,
            #         'existing_vars': [],
            #         'missing_vars': variables,
            #         'needs_merge': False
            #     }
                
            #     if self.store:
            #         cached, existing_vars = self._check_cache_fast(group_cols, variables, query)
            #         combined_item['cached'] = cached
            #         combined_item['existing_vars'] = existing_vars
            #         combined_item['missing_vars'] = [v for v in variables if v not in existing_vars]
            #         combined_item['needs_merge'] = cached and len(existing_vars) > 0 and len(combined_item['missing_vars']) > 0
                
            #     plan.append(combined_item)
        
        return plan

    def _check_cache_fast(self, grouping_spec, variables: List[str], query: Optional[str]) -> tuple[bool, List[str]]:
        """
        Fast cache check using parquet metadata only (no data loading).
        
        Returns:
        --------
        (cached: bool, existing_vars: list of str)
        - cached: True if parquet file exists
        - existing_vars: list of variables already present in the parquet
        """
        if not self.store or not self.store.has(grouping_spec, query):
            return False, []
        
        try:
            # Use dask to read metadata only (no compute)
            ddf = self.store.get(grouping_spec, query)
            if ddf is not None:
                existing_cols = set(ddf.columns)
                existing_vars = [v for v in variables if v in existing_cols]
                return True, existing_vars
        except Exception as e:
            logger.debug(f"Failed to check cache for {grouping_spec}: {e}")
        
        return False, []
    
    def _compute_means_pandas(self, data, variables: List[str], 
                              grouping_plan: List[Dict], query: Optional[str] = None) -> bool:
        """Compute means using pandas groupby (for DataFrame data). Handles grouped or overall means.

        Store parquet files that contain the original grouping columns (no hashed group_key).
        Uses the grouping plan to determine what to compute and whether to merge.
        """
        logger.info("Using pandas for mean computation")
        df = data._dataframe

        for plan_item in grouping_plan:
            # Skip if no missing variables for this grouping
            if not plan_item['missing_vars']:
                logger.info(f"  Skipping {plan_item['type']} grouping {plan_item['cols']}: all variables cached")
                continue
            
            # Only compute missing variables
            vars_to_compute = plan_item['missing_vars']
            
            if plan_item['type'] == '__overall__':
                # Overall means
                means_dict = {var: float(df[var].mean()) for var in vars_to_compute}
                logger.info(f"  overall means: {means_dict}")
                
                if self.store:
                    self._write_or_merge_means(
                        query=query,
                        grouping_spec=plan_item['hash_spec'],
                        data_df=pd.DataFrame([means_dict]),
                        grouping_cols=[],
                        needs_merge=plan_item['needs_merge']
                    )
            
            elif plan_item['type'] in ('__level__', '__combined__'):
                # Grouped means
                cols = plan_item['cols']
                grouped = df.groupby(cols)[vars_to_compute].mean().reset_index()
                out_df = grouped[cols + vars_to_compute].copy()
                logger.info(f"  computed {plan_item['type']} means for {cols}: {len(out_df):,} groups")
                
                if self.store:
                    self._write_or_merge_means(
                        query=query,
                        grouping_spec=plan_item['hash_spec'],
                        data_df=out_df,
                        grouping_cols=cols,
                        needs_merge=plan_item['needs_merge']
                    )

        return True

    def _write_or_merge_means(self, query: Optional[str], grouping_spec, data_df: Union[pd.DataFrame, dd.DataFrame], 
                             grouping_cols: List[str], needs_merge: bool = False):
        """
        Write means to parquet, merging with existing data if needed.
        Uses dask to handle large parquet files that may not fit in memory.
        
        Parameters:
        -----------
        query : str, optional
            Query string for filtering (used for path resolution)
        grouping_spec : list
            Grouping specification for hashing (e.g., ['__overall__'], [['col1', 'col2']], etc.)
        data_df : DataFrame (pandas or dask)
            New data to write (contains grouping cols + variable means)
        grouping_cols : list of str
            List of grouping column names (empty for overall)
        needs_merge : bool
            If True, merge with existing parquet is required
        """
        try:
            store_dir = self.store.ensure_store_path(query=query)
            if store_dir is None:
                return
            
            # Convert pandas DataFrame to dask if necessary
            if isinstance(data_df, pd.DataFrame):
                data_df = dd.from_pandas(data_df, npartitions=1)
            
            ghash = self.store.hash_group_cols(grouping_spec)
            out_path = store_dir / f"{ghash}.parquet"
            
            # Check if merge is needed
            if needs_merge and out_path.exists():
                try:
                    # Read existing parquet with dask (lazy)
                    existing_ddf = dd.read_parquet(str(out_path), engine='pyarrow')
                    
                    # Determine new variable columns
                    new_var_cols = [c for c in data_df.columns if c not in grouping_cols and c not in existing_ddf.columns]
                    
                    if new_var_cols:
                        # Select new data columns
                        new_ddf = data_df[grouping_cols + new_var_cols]
                        
                        # Merge on grouping columns (or handle overall case)
                        if grouping_cols:
                            # Lazy merge
                            merged_ddf = dd.merge(existing_ddf, new_ddf, on=grouping_cols, how='outer')
                        else:
                            # Overall case: add new columns to existing (compute for single row)
                            existing_pdf = existing_ddf.compute()
                            new_pdf = new_ddf.compute()
                            if len(new_pdf) > 0:
                                new_values = new_pdf.iloc[0]
                            else:
                                new_values = {col: np.nan for col in new_var_cols}
                            for col in new_var_cols:
                                existing_pdf[col] = new_values[col]
                            merged_ddf = dd.from_pandas(existing_pdf, npartitions=1)
                        
                        # Write back to parquet (lazy computation happens here)
                        merged_ddf.to_parquet(str(out_path), engine='pyarrow', overwrite=True)
                        logger.info(f"Merged new variables {new_var_cols} into {out_path}")
                    else:
                        logger.info(f"All variables already present in {out_path}")
                except Exception as e:
                    logger.warning(f"Failed to merge with existing parquet at {out_path}, overwriting: {e}")
                    data_df.to_parquet(str(out_path), engine='pyarrow', overwrite=True)
            else:
                # No merge needed or file doesn't exist: write new
                data_df.to_parquet(str(out_path), engine='pyarrow', overwrite=True)
                logger.info(f"Wrote new means to {out_path}")
            
            # Mark store refreshed
            try:
                self.store.refresh()
            except Exception:
                pass
                
        except Exception as e:
            logger.warning(f"Failed to write/merge means parquet: {e}")

    def _all_group_columns(self, levels: list[list[str]]) -> list[str]:
        """Get all unique group columns in consistent order."""
        seen = []
        for level in levels:
            for col in level:
                if col not in seen:
                    seen.append(col)
        return seen

    def _compute_means_duckdb(self, data, variables: List[str], 
                            grouping_plan: List[Dict], query: Optional[str] = None) -> bool:
        try:
            import duckdb
        except ImportError:
            raise ImportError("DuckDB is required for computing means from Parquet files. Install with: pip install duckdb")

        logger.info("Using DuckDB for mean computation")

        if os.path.isdir(data.info.source_path):
            parquet_pattern = f"{data.info.source_path}/**/*.parquet"
        else:
            parquet_pattern = f"{data.info.source_path}"

        where_clause = ""
        if data.query and not data._filters:
            where_clause = f"WHERE {self._query_to_sql(data.query)}"

        # Ensure we have a MeansStore target directory
        if not self.store:
            logger.warning("No MeansStore configured; duckdb path will compute but not persist means")
            store_dir = None
        else:
            store_dir = self.store.ensure_store_path(query=query)

        con = duckdb.connect()
        temp_dir = self.store.get_temp_dir() if self.store else Path(tempfile.mkdtemp(prefix='streamreg_means_'))
        
        try:
            for plan_item in grouping_plan:
                # Skip if no missing variables
                if not plan_item['missing_vars']:
                    logger.info(f"  Skipping {plan_item['type']} grouping: all variables cached")
                    continue
                
                vars_to_compute = plan_item['missing_vars']
                grouping_type = plan_item['type']
                cols_sql = plan_item['cols']
                hash_arg = plan_item['hash_spec']
                
                if grouping_type == '__overall__':
                    select_group_cols = ""
                    group_by_clause = ""
                    grouping_cols = []
                else:
                    cols_quoted = ", ".join(f'"{c}"' for c in cols_sql)
                    select_group_cols = cols_quoted + ", "
                    group_by_clause = f"GROUP BY {cols_quoted}"
                    grouping_cols = cols_sql

                avg_exprs = ", ".join(f'AVG("{v}") AS "{v}"' for v in vars_to_compute)
                sql = f"""
                    SELECT {select_group_cols} {avg_exprs}
                    FROM read_parquet('{parquet_pattern}')
                    {where_clause}
                    {group_by_clause}
                """

                if store_dir is not None:
                    try:
                        # Write to temp location first
                        temp_parquet = Path(temp_dir) / f"temp_{grouping_type}.parquet"
                        copy_sql = f"COPY ({sql}) TO '{str(temp_parquet)}' (FORMAT PARQUET)"
                        logger.info(f"  Running DuckDB query for {grouping_type}")
                        con.execute(copy_sql)
                        
                        # Read temp result and merge/write
                        temp_df = dd.read_parquet(str(temp_parquet), engine='pyarrow')
                        self._write_or_merge_means(
                            query=query,
                            grouping_spec=hash_arg,
                            data_df=temp_df,
                            grouping_cols=grouping_cols,
                            needs_merge=plan_item['needs_merge']
                        )
                        
                        # Clean up temp file
                        temp_parquet.unlink()
                        
                    except Exception as e:
                        logger.warning(f"Failed to compute/write grouping {cols_sql}: {e}")
                else:
                    try:
                        logger.info(f"  Running DuckDB query for grouping {cols_sql} (no store configured)")
                        con.execute(sql).fetchall()
                    except Exception as e:
                        logger.warning(f"Query failed for grouping {cols_sql}: {e}")
        finally:
            con.close()
            # Clean up temp directory
            try:
                if Path(temp_dir).exists():
                    import shutil
                    shutil.rmtree(temp_dir)
            except Exception:
                pass

        return True

    def _query_to_sql(self, query: str) -> str:
        """Convert simple pandas query to SQL WHERE clause."""
        # Simple conversion for basic queries
        sql = query.replace(" and ", " AND ")
        sql = sql.replace(" or ", " OR ")
        sql = sql.replace("==", "=")
        
        # Handle .isin() -> IN
        import re
        sql = re.sub(r'(\w+)\.isin\(\[([^\]]+)\]\)', r'\1 IN (\2)', sql)
        
        return sql

    def _store_grouping_spec(self, group_spec):
        """
        Convert an input grouping specification into the exact shape used when writing means.

        This ensures the hash computed for reading matches the hash computed for writing.

        Mapping:
        - '__overall__' or None -> ['__overall__']
        - 'column' -> [['column']]  (per-level with single column)
        - ['col1', 'col2'] -> ['col1', 'col2']  (combined grouping, multiple top-level items)
        - [['a', 'b']] -> [['a', 'b']]  (per-level with composite key)
        - [['a', 'b'], 'year'] -> [['a', 'b'], 'year']  (combined multi-level)

        Returns:
        --------
        list: Normalized grouping spec matching the format used in _compute_means_pandas
        """
        # Canonicalize group_spec
        canonical = []
        # overall
        if group_spec is None or group_spec == '__overall__':
            canonical = ['__overall__']
        # string -> per-level single column
        elif isinstance(group_spec, str):
            canonical = [[group_spec]]
        # list/tuple handling
        elif isinstance(group_spec, (list, tuple)):
            g = list(group_spec)
            if g == ['__overall__']:
                canonical = ['__overall__']
            elif len(g) == 0:
                canonical = ['__overall__']
            # already a single composite key like [['a','b']]
            elif len(g) == 1 and isinstance(g[0], (list, tuple)):
                canonical = g
            # all strings -> either per-level (single) or combined (multiple)
            elif all(isinstance(item, str) for item in g):
                if len(g) == 1:
                    canonical = [g]
                else:
                    canonical = g
            # mixed case: normalize each element
            else:
                out = []
                for item in g:
                    if isinstance(item, str):
                        out.append(item)
                    elif isinstance(item, (list, tuple)):
                        inner = [str(x) for x in item]
                        out.append(sorted(inner))
                    else:
                        out.append(str(item))
                canonical = out
        # fallback
        else:
            canonical = [[str(group_spec)]]

        return canonical

class DemeanTransformer:
    """
    Apply demeaning transformations using precomputed means (lazy, dask-based).
    
    Unified workflow:
    - Sequential: subtract each level's means in order, then add back overall mean
    - Combined: subtract combined means directly
    
    All operations are lazy using Dask DataFrames.
    """
    
    def __init__(self, source_path: Optional[Path],
                 group_cols: List[Union[str, List[str]]],
                 demean_vars: List[str],
                 sequential: bool = False,
                 query: Optional[str] = None):
        """
        Initialize transformer with means storage access.
        
        Parameters:
        -----------
        source_path : Path or None
            Source path of the dataset for means storage
        group_cols : list of str/list
            Grouping specification
        demean_vars : list of str
            Variables to demean
        sequential : bool
            If True, apply demeaning sequentially across levels
        query : str, optional
            Query used when computing means (for cache key)
        """
        self.source_path = source_path
        self.group_cols = group_cols or []
        self.demean_vars = demean_vars
        self.sequential = sequential
        self.query = query
        
        # Only allow up to 2-way for now
        if len(self.group_cols) > 2:
            raise NotImplementedError("Only up to 2-way demeaning is supported (3-way+ is not implemented).")
        
        # Initialize MeansStore (lazy)
        self._store = None
        if source_path:
            from streamreg.means_store import MeansStore
            self._store = MeansStore(source_path)

    def transform(self, ddf):
        """
        Apply demeaning to a Dask DataFrame (lazy operation).
        
        Parameters:
        -----------
        ddf : dask.DataFrame or pd.DataFrame
            Input data
        
        Returns:
        --------
        dask.DataFrame with demeaning applied (uncomputed)
        """
        if self._store is None:
            logger.warning("No MeansStore configured; returning input unchanged")
            return ddf if isinstance(ddf, dd.DataFrame) else dd.from_pandas(ddf, npartitions=1)
        
        # Ensure input is Dask DataFrame
        if isinstance(ddf, pd.DataFrame):
            ddf = dd.from_pandas(ddf, npartitions=1)
        
        if self.sequential:
            return self._apply_sequential_demeaning(ddf)
        else:
            return self._apply_combined_demeaning(ddf)
    
    def _apply_sequential_demeaning(self, ddf):
        """
        Sequential demeaning: subtract each level's means, then add back overall.
        All operations are lazy.
        
        For each level:
        1. Merge level means
        2. Subtract from variables
        
        Finally:
        3. Merge overall means
        4. Add back to variables
        """
        result = ddf
        
        # Subtract means for each level
        for level_spec in self.group_cols:
            # Pass level_spec wrapped in list to match how it was written
            result = self._merge_and_subtract(result, level_spec)
        
        # Add back overall mean
        result = self._merge_and_add(result, '__overall__')
        
        return result
    
    def _apply_combined_demeaning(self, ddf):
        """
        Combined demeaning: subtract combined group means directly.
        All operations are lazy.
        """
        # Use full group_cols spec to fetch combined means
        return self._merge_and_subtract(ddf, self.group_cols)
    
    def _merge_and_subtract(self, ddf, group_spec):
        """
        Lazy merge with means and subtract from variables.
        
        Parameters:
        -----------
        ddf : dask.DataFrame
            Input data
        group_spec : str, list, or list of lists
            Group specification for means lookup
        
        Returns:
        --------
        dask.DataFrame with means subtracted (lazy)
        """
        
        # Load means (returns Dask DataFrame)
        means_ddf = self._load_means(group_spec)
        if means_ddf is None:
            logger.warning(f"No means found for grouping {group_spec}; skipping")
            return ddf
        
        # Standard case: merge on grouping columns
        # Rename mean columns to avoid collision
        mean_cols = {v: f"{v}__mean" for v in self.demean_vars if v in means_ddf.columns}
        means_ddf = means_ddf.rename(columns=mean_cols)
        
        # Select only merge columns + renamed mean columns
        select_cols = list(mean_cols.values())
        means_ddf = means_ddf[select_cols]
        
        # Lazy merge
        merged = dd.merge(ddf, means_ddf, left_on=group_spec, right_index=True, how='left')
        
        # Lazy subtraction
        for var, mean_col in mean_cols.items():
            if var in merged.columns and mean_col in merged.columns:
                merged = merged.assign(**{var: merged[var] - merged[mean_col].fillna(0.0)})
        
        # Drop mean columns
        merged = merged.drop(columns=list(mean_cols.values()))
        
        return merged
    
    def _merge_and_add(self, ddf, group_spec):
        """
        Lazy merge with means and add to variables (for adding back overall mean).
        
        Parameters:
        -----------
        ddf : dask.DataFrame
            Input data
        group_spec : str or list
            Group specification for means lookup (typically '__overall__')
        
        Returns:
        --------
        dask.DataFrame with means added (lazy)
        """
        # Load means (returns Dask DataFrame)
        means_ddf = self._load_means(group_spec)
        if means_ddf is None:
            logger.warning(f"No means found for grouping {group_spec}; skipping")
            return ddf
        
        # Overall means: merge on constant and add
        mean_cols = {v: f"{v}__mean" for v in self.demean_vars if v in means_ddf.columns}
        means_ddf = means_ddf.rename(columns=mean_cols)
        
        # Add constant merge key
        const_key = "_sr_overall_key"
        ddf = ddf.assign(**{const_key: 0})
        means_ddf = means_ddf.assign(**{const_key: 0})
        
        # Select only constant key + renamed mean columns
        select_cols = [const_key] + list(mean_cols.values())
        means_ddf = means_ddf[select_cols]
        
        # Lazy merge
        merged = dd.merge(ddf, means_ddf, on=const_key, how='left')
        
        # Lazy addition
        for var, mean_col in mean_cols.items():
            if var in merged.columns and mean_col in merged.columns:
                merged = merged.assign(**{var: merged[var] + merged[mean_col].fillna(0.0)})
        
        # Drop helper columns
        merged = merged.drop(columns=[const_key] + list(mean_cols.values()))
        
        return merged
    
    def _merge_on_constant_and_subtract(self, ddf, means_ddf):
        """Helper for overall case: merge on constant key and subtract."""
        const_key = "_sr_overall_key"
        
        # Rename mean columns
        mean_cols = {v: f"{v}__mean" for v in self.demean_vars if v in means_ddf.columns}
        means_ddf = means_ddf.rename(columns=mean_cols)
        
        # Add constant key to both
        ddf = ddf.assign(**{const_key: 0})
        means_ddf = means_ddf.assign(**{const_key: 0})
        
        # Select only constant key + mean columns
        select_cols = [const_key] + list(mean_cols.values())
        means_ddf = means_ddf[select_cols]
        
        # Lazy merge
        merged = dd.merge(ddf, means_ddf, on=const_key, how='left')
        
        # Lazy subtraction
        for var, mean_col in mean_cols.items():
            if var in merged.columns and mean_col in merged.columns:
                merged = merged.assign(**{var: merged[var] - merged[mean_col].fillna(0.0)})
        
        # Drop helper columns
        merged = merged.drop(columns=[const_key] + list(mean_cols.values()))
        
        return merged
    
    def _load_means(self, group_spec):
        """
        Load means from storage as Dask DataFrame (lazy).
        
        Parameters:
        -----------
        group_spec : list
            Group specification in normalized format
        
        Returns:
        --------
        dask.DataFrame or None
        """
        if not self.demean_vars:
            return None
        
        # Use store's get method with query parameter
        means_ddf = self._store.get(group_spec, query=self.query)
        
        if means_ddf is None:
            return None
        
        # Verify all required variables are present
        missing = [v for v in self.demean_vars if v not in means_ddf.columns]
        if missing:
            logger.warning(f"Means parquet missing variables: {missing}")
        
        return means_ddf
    
    def _get_merge_columns(self, means_ddf, group_spec):
        """
        Determine which columns to use for merging based on means parquet structure.
        
        Parameters:
        -----------
        means_ddf : dask.DataFrame
            Means data
        group_spec : list
            Group specification
        
        Returns:
        --------
        list of str: column names to merge on (empty for overall case)
        """
        if group_spec == ['__overall__']:
            return []
        
        # Extract grouping columns from means parquet
        # (exclude variable columns)
        means_cols = set(means_ddf.columns)
        var_cols = set(self.demean_vars)
        
        merge_cols = [col for col in means_cols if col not in var_cols]
        
        return merge_cols