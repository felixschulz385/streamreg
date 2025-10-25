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
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DemeanComputer:
    """
    Compute group means from streaming data using pandas or DuckDB.
    """
    
    def __init__(self, dataset_root: Optional[Path] = None):
        """
        Initialize computer.
        
        Parameters:
        -----------
        dataset_root : Path, optional
            Root directory of dataset for means storage. If None, no caching.
        """
        self.dataset_root = dataset_root
        self.store = None
        
        if dataset_root:
            from streamreg.means_store import MeansStore
            self.store = MeansStore(dataset_root)
            self.store.invalidate_if_stale()
    
    def compute_means(self, data, variables: List[str], group_cols: Union[str, List[Union[str, List[str]]]],
                      query: Optional[str] = None, variable_batch_size: int = 50) -> Dict[str, pd.Series]:
        """
        Compute group means for multiple variables.
        
        Parameters:
        -----------
        data : StreamData
            Data source
        variables : list of str
            Variables to compute means for
        group_cols : str or list of str/list
            Grouping specification. Examples:
            - 'country': single column
            - ['country', 'year']: two separate columns
            - [['tile_ix', 'tile_iy', 'pixel_id'], 'year']: multi-level with combined key
        query : str, optional
            Query used for filtering (for cache key)
        variable_batch_size : int
            Maximum variables to compute per batch (to manage memory)
        
        Returns:
        --------
        dict: {variable: Series of group means}
        """
        # Normalize group_cols to list
        if isinstance(group_cols, str):
            group_cols = [group_cols]
        
        # Flatten for display
        flat_display = self._flatten_group_cols(group_cols)
        logger.info(f"Computing group means for {len(variables)} variables by {flat_display}")
        
        # Check cache for each variable
        means_dict = {}
        variables_to_compute = []
        
        if self.store:
            for var in variables:
                cached_means = self.store.get(var, group_cols, query)
                if cached_means is not None:
                    means_dict[var] = cached_means
                else:
                    variables_to_compute.append(var)
        else:
            variables_to_compute = variables
        
        if not variables_to_compute:
            logger.info("All group means found in cache")
            return means_dict
        
        logger.info(f"Computing means for {len(variables_to_compute)} variables")
        
        # Process variables in batches if many variables
        if len(variables_to_compute) > variable_batch_size:
            logger.info(f"Processing {len(variables_to_compute)} variables in batches of {variable_batch_size}")
            for i in range(0, len(variables_to_compute), variable_batch_size):
                batch = variables_to_compute[i:i+variable_batch_size]
                
                # Compute means for batch based on data type
                if data.info.source_type == 'dataframe':
                    batch_means = self._compute_pandas(data, batch, group_cols)
                else:
                    batch_means = self._compute_duckdb(data, batch, group_cols)
                
                # Cache and merge batch results
                for var, means in batch_means.items():
                    if self.store:
                        self.store.put(var, group_cols, means, query)
                    means_dict[var] = means
        else:
            # Single batch: compute all variables at once
            if data.info.source_type == 'dataframe':
                computed_means = self._compute_pandas(data, variables_to_compute, group_cols)
            else:
                computed_means = self._compute_duckdb(data, variables_to_compute, group_cols)
            
            # Cache and merge results
            for var, means in computed_means.items():
                if self.store:
                    self.store.put(var, group_cols, means, query)
                means_dict[var] = means
        
        return means_dict
    
    def compute_means_sequential(self, data, variables: List[str], 
                                group_levels: List[Union[str, List[str]]],
                                query: Optional[str] = None,
                                variable_batch_size: int = 50) -> Dict[str, List[pd.Series]]:
        """
        Compute group means separately for each grouping level (for sequential demeaning).
        
        Parameters:
        -----------
        data : StreamData
            Data source
        variables : list of str
            Variables to compute means for
        group_levels : list of str/list
            List of grouping levels, each computed separately.
            Example: [['tile_ix', 'tile_iy', 'pixel_id'], 'year']
            Will compute means by tiles (across all years), then by year (across all tiles)
        query : str, optional
            Query used for filtering
        variable_batch_size : int
            Maximum variables to compute per batch
        
        Returns:
        --------
        dict: {variable: [Series_level1, Series_level2, ..., Series_overall]}
        """
        means_by_level = {var: [] for var in variables}
        
        # Compute means for each level independently
        for level_idx, group_spec in enumerate(group_levels):
            logger.info(f"Computing means for level {level_idx + 1}/{len(group_levels)}")
            
            # Compute means for this level only (not cross-product)
            level_means = self.compute_means(
                data, 
                variables, 
                group_cols=[group_spec],  # Single level
                query=query,
                variable_batch_size=variable_batch_size
            )
            
            # Append to results
            for var in variables:
                means_by_level[var].append(level_means[var])
        
        # Compute overall mean (grand mean) for inclusion-exclusion
        logger.info("Computing overall means")
        overall_means = self._compute_overall_means(data, variables, query)
        
        # Cache overall means with special key
        if self.store:
            for var in variables:
                self.store.put(var, ['__overall__'], overall_means[var], query)
        
        for var in variables:
            means_by_level[var].append(overall_means[var])
        
        return means_by_level
    
    def _flatten_group_cols(self, group_cols: List[Union[str, List[str]]]) -> List[str]:
        """Flatten group_cols for display or storage."""
        flat = []
        for item in group_cols:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat
    
    def _create_combined_key(self, df: pd.DataFrame, group_spec: Union[str, List[str]]) -> pd.Series:
        """
        Create a combined key from a group specification.
        
        Parameters:
        -----------
        df : DataFrame
            Data containing the columns
        group_spec : str or list of str
            Single column name or list of column names to combine
        
        Returns:
        --------
        Series with combined keys
        """
        if isinstance(group_spec, str):
            return df[group_spec].astype(str)
        else:
            # Combine multiple columns with underscore separator
            return df[group_spec].apply(
                lambda row: '_'.join(row.astype(str)), axis=1
            )
    
    def _compute_pandas(self, data, variables: List[str], 
                       group_cols: List[Union[str, List[str]]]) -> Dict[str, pd.Series]:
        """Compute means using pandas groupby (for DataFrame data)."""
        logger.info("Using pandas groupby for mean computation")
        
        # Get DataFrame (already filtered by query if provided)
        df = data._dataframe
        
        # Create combined keys for each group level
        group_keys = []
        group_labels = []
        
        for i, group_spec in enumerate(group_cols):
            key_col = f'_group_key_{i}'
            df[key_col] = self._create_combined_key(df, group_spec)
            group_keys.append(key_col)
            
            if isinstance(group_spec, list):
                group_labels.append('+'.join(group_spec))
            else:
                group_labels.append(group_spec)
        
        # Compute means using the combined keys (each key_col is now already a combined key)
        grouped = df.groupby(group_keys)[variables].mean()
        
        # Convert to dict of Series with combined index
        means_dict = {}
        for var in variables:
            means = grouped[var].copy()
            
            # Create combined index string using pipe separator
            if len(group_keys) == 1:
                # Single level - use as is (may already be combined from multiple columns)
                means.index = means.index.astype(str)
            else:
                # Multi-level - combine with pipe separator
                # Note: each level may itself be a combined key (e.g., "tile1_tile2_pixel")
                means.index = means.index.map(lambda x: '|'.join(map(str, x)) if isinstance(x, tuple) else str(x))
            
            means_dict[var] = means
            logger.info(f"  {var}: {len(means):,} groups, mean={means.mean():.4f}")
        
        # Clean up temporary columns
        for key_col in group_keys:
            df.drop(columns=[key_col], inplace=True)
        
        return means_dict
    
    def _compute_duckdb(self, data, variables: List[str], 
                       group_cols: List[Union[str, List[str]]]) -> Dict[str, pd.Series]:
        """Compute means using DuckDB (for Parquet data) - BATCH ALL VARIABLES."""
        try:
            import duckdb
        except ImportError:
            raise ImportError("DuckDB is required for computing means from Parquet files. "
                            "Install with: pip install duckdb")
        
        logger.info("Using DuckDB for mean computation")
        
        # Get parquet file path(s)
        if data.info.source_type == 'parquet':
            parquet_pattern = f"{data.info.source_path}"
        elif data.info.source_type == 'partitioned':
            parquet_pattern = f"{data.info.source_path}/**/*.parquet"
        else:
            raise ValueError(f"Unsupported source type for DuckDB: {data.info.source_type}")
        
        # Build combined key expressions for each group level
        # Each level can be a single column OR a combined key from multiple columns
        key_exprs = []
        group_by_cols = []  # Flat list of columns for GROUP BY
        
        for i, group_spec in enumerate(group_cols):
            if isinstance(group_spec, list):
                # Multi-column group: create combined key with underscore separator
                parts = " || '_' || ".join([f"CAST({col} AS VARCHAR)" for col in group_spec])
                key_exprs.append(f"({parts}) AS _group_key_{i}")
                # For GROUP BY, we need the individual columns
                group_by_cols.extend(group_spec)
            else:
                # Single column group
                key_exprs.append(f"CAST({group_spec} AS VARCHAR) AS _group_key_{i}")
                group_by_cols.append(group_spec)
        
        # Add WHERE clause for query filter
        where_clause = ""
        if data.query and not data._filters:
            where_clause = f"WHERE {self._query_to_sql(data.query)}"
        
        # **OPTIMIZATION**: Compute ALL variables in a SINGLE query
        # Build list of AVG expressions for all variables
        avg_exprs = [f"AVG({var}) AS {var}_mean" for var in variables]
        
        # Build the query differently based on number of grouping levels
        if len(key_exprs) == 1:
            # Single group level: simpler query
            query = f"""
            SELECT 
                {key_exprs[0].replace(f' AS _group_key_0', ' AS _group_key')},
                {', '.join(avg_exprs)}
            FROM read_parquet('{parquet_pattern}')
            {where_clause}
            GROUP BY {', '.join(group_by_cols)}
            """
        else:
            # Multiple group levels: need intermediate CTE to compute keys, then combine
            # First CTE computes individual level keys
            pipe_separator = " || '|' || "
            group_key_expr = pipe_separator.join([f'_group_key_{i}' for i in range(len(key_exprs))])
            
            query = f"""
            WITH level_keys AS (
                SELECT 
                    {', '.join(key_exprs)},
                    {', '.join(avg_exprs)}
                FROM read_parquet('{parquet_pattern}')
                {where_clause}
                GROUP BY {', '.join(group_by_cols)}
            )
            SELECT 
                {group_key_expr} AS _group_key,
                {', '.join([f'{var}_mean' for var in variables])}
            FROM level_keys
            """
        
        logger.debug(f"DuckDB query for {len(variables)} variables (single scan)")
        
        # Execute once and get all results
        con = duckdb.connect()
        means_dict = {}
        
        try:
            result_df = con.execute(query).df()
            
            # Split result into separate Series for each variable
            group_keys = result_df['_group_key'].values
            
            for var in variables:
                means = pd.Series(
                    result_df[f'{var}_mean'].values,
                    index=group_keys,
                    name=var,
                    copy=False  # No copy, use view
                )
                means_dict[var] = means
                logger.info(f"  {var}: {len(means):,} groups, mean={means.mean():.4f}")
            
            # Clear result to free memory
            del result_df
            
        finally:
            con.close()
        
        return means_dict
    
    def _compute_overall_means(self, data, variables: List[str], 
                              query: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Compute overall (grand) means for variables.
        
        Returns a dict with single-row Series (index=['overall']).
        """
        overall_dict = {}
        
        if data.info.source_type == 'dataframe':
            df = data._dataframe
            for var in variables:
                mean_val = df[var].mean()
                overall_dict[var] = pd.Series([mean_val], index=['overall'], name=var)
        else:
            # Use DuckDB for parquet
            try:
                import duckdb
            except ImportError:
                raise ImportError("DuckDB is required for computing means from Parquet files.")
            
            if data.info.source_type == 'parquet':
                parquet_pattern = f"{data.info.source_path}"
            elif data.info.source_type == 'partitioned':
                parquet_pattern = f"{data.info.source_path}/**/*.parquet"
            else:
                raise ValueError(f"Unsupported source type: {data.info.source_type}")
            
            where_clause = ""
            if data.query and not data._filters:
                where_clause = f"WHERE {self._query_to_sql(data.query)}"
            
            avg_exprs = [f"AVG({var}) AS {var}_mean" for var in variables]
            query_str = f"""
            SELECT {', '.join(avg_exprs)}
            FROM read_parquet('{parquet_pattern}')
            {where_clause}
            """
            
            con = duckdb.connect()
            try:
                result_df = con.execute(query_str).df()
                for var in variables:
                    mean_val = result_df[f'{var}_mean'].iloc[0]
                    overall_dict[var] = pd.Series([mean_val], index=['overall'], name=var)
            finally:
                con.close()
        
        return overall_dict
    
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


class DemeanTransformer:
    """
    Apply demeaning transformations using precomputed means (memory-efficient).
    
    Supports:
    - Single-level demeaning: demean by one grouping variable
    - Multi-level sequential demeaning: uses inclusion-exclusion principle
      * 2-way: var_dm = var - mean_dim1 - mean_dim2 + mean_overall
      * 3-way: var_dm = var - sum(single_dim_means) + sum(pairwise_means) - mean_overall
    
    Workers query LMDB on-demand for only the groups present in their chunk.
    """
    
    def __init__(self, dataset_root: Optional[Path],
                 group_cols: List[Union[str, List[str]]],
                 demean_vars: List[str],
                 sequential: bool = False,
                 query: Optional[str] = None):
        """
        Initialize transformer with LMDB access.
        
        Parameters:
        -----------
        dataset_root : Path or None
            Root directory for LMDB means storage
        group_cols : list of str/list
            Grouping specification
        demean_vars : list of str
            Variables to demean
        sequential : bool
            If True, apply demeaning sequentially across levels
        query : str, optional
            Query used when computing means (for cache key)
        """
        self.dataset_root = dataset_root
        self.group_cols = group_cols
        self.demean_vars = demean_vars
        self.sequential = sequential
        self.query = query
        
        # Initialize LMDB store (lazy, opens on first use)
        self._store = None
        if dataset_root:
            from streamreg.means_store import MeansStore
            self._store = MeansStore(dataset_root)
    
    def _create_combined_key(self, df: pd.DataFrame, group_spec: Union[str, List[str]]) -> pd.Series:
        """Create a combined key from a group specification."""
        if isinstance(group_spec, str):
            return df[group_spec].astype(str)
        else:
            return df[group_spec].apply(
                lambda row: '_'.join(row.astype(str)), axis=1
            )
    
    def _get_means_for_groups(self, var: str, group_spec: Union[str, List[str]], 
                              unique_groups: np.ndarray) -> pd.Series:
        """
        Query LMDB for means of specific groups only.
        
        Parameters:
        -----------
        var : str
            Variable name
        group_spec : str or list of str
            Grouping specification
        unique_groups : array
            Unique group keys present in chunk
        
        Returns:
        --------
        Series with means for requested groups only
        """
        if self._store is None:
            logger.warning(f"No means store available, cannot demean {var}")
            return pd.Series(0.0, index=unique_groups)
        
        # Get full means from LMDB (cached in memory per variable)
        group_cols_normalized = [group_spec] if isinstance(group_spec, (str, list)) else group_spec
        full_means = self._store.get(var, group_cols_normalized, self.query)
        
        if full_means is None:
            logger.warning(f"No cached means found for {var}, skipping demeaning")
            return pd.Series(0.0, index=unique_groups)
        
        # Extract only the groups present in this chunk
        chunk_means = full_means.reindex(unique_groups, fill_value=0.0)
        
        return chunk_means
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply demeaning to a DataFrame chunk.
        
        Parameters:
        -----------
        df : DataFrame
            Chunk to demean
        
        Returns:
        --------
        DataFrame with demeaned variables
        """
        if self.sequential:
            return self._transform_sequential(df)
        else:
            return self._transform_combined(df)
    
    def _transform_sequential(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply demeaning using inclusion-exclusion principle for multi-level."""
        df = df.copy()
        
        # For each variable, apply inclusion-exclusion formula
        for var in self.demean_vars:
            if var not in df.columns:
                continue
            
            n_levels = len(self.group_cols)
            
            # Start with original value
            result = df[var].copy()
            
            # Subtract single-dimension means (first n_levels terms)
            for level_idx in range(n_levels):
                group_spec = self.group_cols[level_idx]
                group_key = self._create_combined_key(df, group_spec)
                
                # Get unique groups in this chunk
                unique_groups = group_key.unique()
                
                # Query LMDB for only these groups
                level_means = self._get_means_for_groups(var, group_spec, unique_groups)
                
                # Map to rows
                var_means = group_key.map(level_means).fillna(0)
                result = result - var_means
                del var_means
            
            # Add back overall mean (last term in means_list)
            # Query overall mean separately (single value, group='overall')
            if self._store is not None:
                # Overall mean stored with special key
                overall_series = self._store.get(var, ['__overall__'], self.query)
                if overall_series is not None and len(overall_series) > 0:
                    overall_mean = overall_series.iloc[0]
                    if n_levels == 2:
                        result = result + overall_mean
                    elif n_levels > 2:
                        logger.warning(f"Inclusion-exclusion for {n_levels}-way demeaning not fully implemented")
                        result = result + (n_levels - 1) * overall_mean
            
            df[var] = result
        
        return df
    
    def _transform_combined(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply demeaning using combined multi-level keys (original behavior)."""
        df = df.copy()
        
        # Create combined key for all group levels
        if len(self.group_cols) == 1:
            group_key = self._create_combined_key(df, self.group_cols[0])
            unique_groups = group_key.unique()
            
            # Demean each variable
            for var in self.demean_vars:
                if var not in df.columns:
                    continue
                
                # Query LMDB for only the groups in this chunk
                means = self._get_means_for_groups(var, self.group_cols[0], unique_groups)
                
                var_means = group_key.map(means).fillna(0)
                df[var] = df[var] - var_means
                del var_means
        else:
            # Multi-level: combine with pipe separator
            key_parts = []
            for group_spec in self.group_cols:
                key_parts.append(self._create_combined_key(df, group_spec))
            group_key = pd.Series(
                ['|'.join(parts) for parts in zip(*key_parts)],
                index=df.index
            )
            unique_groups = group_key.unique()
            
            # Demean each variable
            for var in self.demean_vars:
                if var not in df.columns:
                    continue
                
                # Query LMDB for combined keys
                means = self._get_means_for_groups(var, self.group_cols, unique_groups)
                
                var_means = group_key.map(means).fillna(0)
                df[var] = df[var] - var_means
                del var_means
        
        return df
