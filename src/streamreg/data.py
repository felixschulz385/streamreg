import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Iterator, Tuple
from dataclasses import dataclass
import logging
import pyarrow.parquet as pq
import re

logger = logging.getLogger(__name__)

# Try to import fastparquet for more efficient filtering
try:
    import fastparquet as fp
    HAS_FASTPARQUET = True
except ImportError:
    HAS_FASTPARQUET = False
    logger.debug("fastparquet not available, using pyarrow for parquet reading")


@dataclass
class DatasetInfo:
    """Metadata about a dataset."""
    n_rows: int
    n_cols: int
    columns: List[str]
    numeric_columns: List[str]
    source_type: str  # 'parquet', 'dataframe', 'partitioned'
    source_path: Optional[Path] = None
    partitions: Optional[List[Path]] = None


class StreamData:
    """
    Unified data interface for streaming regression with efficient subsetting.
    
    Supports:
    - Pandas DataFrame (in-memory, filtered once at initialization)
    - Single parquet file (with filter pushdown when possible)
    - Partitioned parquet dataset (with partition pruning and filter pushdown)
    
    All parallel processing is handled internally - users don't need to know about partitioning.
    
    Performance optimizations:
    - For DataFrames: Query applied once at initialization, then chunked
    - For Parquet: Query converted to filters for pushdown when possible, otherwise applied after read
    - For Partitioned Parquet: Partition pruning before reading
    - Only loads columns needed for modeling, drops filter-only columns after filtering
    """
    
    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        chunk_size: int = 10000,
        query: Optional[str] = None,
        backend: str = 'auto'
    ):
        """
        Initialize data source with efficient filtering.
        
        Parameters:
        -----------
        data : str, Path, or DataFrame
            Data source to load
        chunk_size : int
            Size of chunks for iteration
        query : str, optional
            Pandas query string to filter data (e.g., "year >= 2000 and country == 'USA'").
            For DataFrames: Applied once at initialization.
            For Parquet files: Automatically converted to filter pushdown when possible.
            Examples:
            - "year >= 2000"
            - "country == 'USA' and year >= 2000"
            - "gdp > 10000"
            - "country.isin(['USA', 'CAN', 'MEX'])"
        backend : str
            Parquet backend: 'auto', 'fastparquet', 'pyarrow'
            'auto' uses fastparquet if available, otherwise pyarrow
        """
        self.chunk_size = chunk_size
        self.query = query
        self.backend = backend
        
        # Internal: converted filters for Parquet pushdown
        self._filters = None
        
        # Determine actual backend for Parquet
        self._parquet_backend = self._select_parquet_backend(backend)
        
        # Compiled filter objects for reuse
        self._compiled_query = None
        self._filter_columns = set()
        
        self._setup_data_source(data)
        
        # Validate and compile query if provided
        if self.query:
            self._validate_and_compile_filters()
    
    def _select_parquet_backend(self, backend: str) -> str:
        """Select the actual Parquet backend to use."""
        if backend == 'pyarrow':
            return 'pyarrow'
        elif backend == 'fastparquet':
            if not HAS_FASTPARQUET:
                logger.warning("fastparquet requested but not available, falling back to pyarrow")
                return 'pyarrow'
            return 'fastparquet'
        else:  # auto
            return 'fastparquet' if HAS_FASTPARQUET else 'pyarrow'
    
    def _validate_and_compile_filters(self):
        """Validate query and compile for reuse."""
        if self.info.source_type == 'dataframe':
            # For DataFrames, validate query
            if self.query:
                try:
                    # Test query on sample
                    sample_df = self._dataframe.head(min(100, len(self._dataframe)))
                    _ = sample_df.query(self.query)
                    logger.debug(f"Query validated: {self.query}")
                except Exception as e:
                    raise ValueError(f"Invalid query string '{self.query}': {e}")
        else:
            # For Parquet files, try to convert query to filters for pushdown
            if self.query:
                self._filters = self._query_to_filters(self.query)
                if self._filters:
                    self._filter_columns = {f[0] for f in self._filters}
                    logger.debug(f"Converted query to filters for pushdown: {self._filters}")
                else:
                    logger.debug(f"Query will be applied after read (not convertible to filters): {self.query}")
    
    def _query_to_filters(self, query: str) -> Optional[List[Tuple]]:
        """
        Convert simple pandas query to fastparquet filters.
        
        Supports basic comparisons like:
        - year >= 2000
        - country == 'USA'
        - year >= 2000 and country == 'USA'
        
        Returns None if query cannot be converted (will fall back to pandas filtering).
        """
        if not query or self._parquet_backend != 'fastparquet':
            return None
        
        # Simple regex-based parser for common query patterns
        filters = []
        
        # Split by 'and' (not supporting 'or' for now as it's complex for filters)
        if ' or ' in query.lower():
            return None  # Cannot convert OR queries to simple filters
        
        parts = [p.strip() for p in re.split(r'\s+and\s+', query, flags=re.IGNORECASE)]
        
        for part in parts:
            # Match patterns like: column op value
            # Operators: ==, !=, <=, >=, <, >, in, not in
            match = re.match(r'(\w+)\s*(==|!=|<=|>=|<|>)\s*(.+)', part.strip())
            if match:
                col, op, val = match.groups()
                col = col.strip()
                val = val.strip()
                
                # Try to evaluate the value
                try:
                    # Handle strings with quotes
                    if (val.startswith("'") and val.endswith("'")) or \
                       (val.startswith('"') and val.endswith('"')):
                        val = val[1:-1]
                    else:
                        # Try to convert to number
                        if '.' in val:
                            val = float(val)
                        else:
                            val = int(val)
                    
                    filters.append((col, op, val))
                except:
                    return None  # Cannot parse value
            else:
                # Check for .isin() pattern
                match_isin = re.match(r'(\w+)\.isin\(\[(.+)\]\)', part.strip())
                if match_isin:
                    col, values_str = match_isin.groups()
                    try:
                        # Parse list of values
                        values = eval(f"[{values_str}]")
                        filters.append((col, 'in', values))
                    except:
                        return None
                else:
                    return None  # Cannot parse this condition
        
        return filters if filters else None
    
    def _extract_query_columns(self) -> List[str]:
        """
        Extract column names referenced in the query string.
        
        Returns:
        --------
        List of column names found in the query
        """
        if self._filters:
            return list(self._filter_columns)
        
        if not self.query:
            return []
        
        # Get all column names from the dataset
        all_columns = set(self.info.columns)
        
        # Find all identifiers in the query that match column names
        potential_cols = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', self.query)
        
        # Filter to only include actual column names
        query_columns = [col for col in potential_cols if col in all_columns]
        
        return list(set(query_columns))
    
    def _setup_data_source(self, data: Union[str, Path, pd.DataFrame]):
        """Setup data source and extract metadata."""
        if isinstance(data, pd.DataFrame):
            self._setup_dataframe(data)
        else:
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Data source not found: {path}")
            
            if path.is_dir():
                self._setup_partitioned_parquet(path)
            elif path.suffix == '.parquet':
                self._setup_single_parquet(path)
            else:
                raise ValueError(f"Unsupported data source: {path}")
    
    def _setup_dataframe(self, df: pd.DataFrame):
        """Setup from DataFrame with immediate filtering."""
        # Apply query immediately for DataFrames
        if self.query:
            try:
                df = df.query(self.query)
                logger.info(f"DataFrame filtered: {len(df):,} rows remaining")
            except Exception as e:
                raise ValueError(f"Invalid query string '{self.query}': {e}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.info = DatasetInfo(
            n_rows=len(df),
            n_cols=len(df.columns),
            columns=df.columns.tolist(),
            numeric_columns=numeric_cols,
            source_type='dataframe'
        )
        self._dataframe = df
        self._parquet_file = None
        
        logger.debug(f"Loaded DataFrame: {self.info.n_rows:,} rows, {self.info.n_cols} columns")
    
    def _filters_to_mask(self, df: pd.DataFrame, filters: List[Tuple]) -> pd.Series:
        """Convert filters to pandas boolean mask."""
        mask = pd.Series(True, index=df.index)
        
        for col, op, val in filters:
            if op == '==':
                mask &= (df[col] == val)
            elif op == '!=':
                mask &= (df[col] != val)
            elif op == '<':
                mask &= (df[col] < val)
            elif op == '<=':
                mask &= (df[col] <= val)
            elif op == '>':
                mask &= (df[col] > val)
            elif op == '>=':
                mask &= (df[col] >= val)
            elif op == 'in':
                mask &= df[col].isin(val)
            elif op == 'not in':
                mask &= ~df[col].isin(val)
        
        return mask
    
    def _setup_single_parquet(self, path: Path):
        """Setup from single parquet file."""
        if self._parquet_backend == 'fastparquet':
            self._parquet_file = fp.ParquetFile(path)
            schema = self._parquet_file.columns
            columns = list(schema)
        else:
            self._parquet_file = pq.ParquetFile(path)
            schema = self._parquet_file.metadata.schema
            columns = [field.name for field in schema]
        
        # Read small sample to determine numeric columns
        sample_df = self._read_sample_parquet(self._parquet_file, 100)
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_rows = self._get_parquet_row_count(self._parquet_file)
        
        self.info = DatasetInfo(
            n_rows=n_rows,
            n_cols=len(columns),
            columns=columns,
            numeric_columns=numeric_cols,
            source_type='parquet',
            source_path=path
        )
        self._dataframe = None
        
        logger.debug(f"Loaded parquet ({self._parquet_backend}): {self.info.n_rows:,} rows, {self.info.n_cols} columns")
    
    def _setup_partitioned_parquet(self, path: Path):
        """Setup from partitioned parquet dataset with partition pruning."""
        partitions = self._discover_and_prune_partitions(path)
        
        # Read first partition for schema
        if self._parquet_backend == 'fastparquet':
            first_parquet = fp.ParquetFile(partitions[0])
            columns = list(first_parquet.columns)
        else:
            first_parquet = pq.ParquetFile(partitions[0])
            schema = first_parquet.metadata.schema
            columns = [field.name for field in schema]
        
        # Estimate total rows from remaining partitions
        total_rows = sum(
            self._get_parquet_row_count(self._open_parquet_file(p))
            for p in partitions
        )
        
        # Read small sample to determine numeric columns
        sample_df = self._read_sample_parquet(first_parquet, 100)
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.info = DatasetInfo(
            n_rows=total_rows,
            n_cols=len(columns),
            columns=columns,
            numeric_columns=numeric_cols,
            source_type='partitioned',
            source_path=path,
            partitions=partitions
        )
        self._dataframe = None
        self._parquet_file = None
        
        logger.info(f"Loaded partitioned dataset ({self._parquet_backend}): {total_rows:,} rows, {len(partitions)} partitions")
    
    def _discover_and_prune_partitions(self, path: Path) -> List[Path]:
        """Discover partitions and prune based on query (Hive-style partitioning)."""
        logger.debug(f"Discovering partitions in {path}")
        
        # Find all .parquet files recursively
        all_files = list(path.rglob("*.parquet"))
        
        if not all_files:
            raise ValueError(f"No parquet files found in {path}")
        
        # If no filters (from query conversion), validate and return all partitions
        if not self._filters:
            return self._validate_partitions(all_files)
        
        # Try to prune based on Hive-style partitioning
        # Format: .../key=value/.../*.parquet
        pruned_files = []
        for file in all_files:
            # Extract partition values from path
            partition_values = {}
            for part in file.parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    # Try to convert to appropriate type
                    try:
                        if value.isdigit():
                            value = int(value)
                        elif value.replace('.', '', 1).isdigit():
                            value = float(value)
                    except:
                        pass
                    partition_values[key] = value
            
            # Check if partition matches filters
            if self._partition_matches_filters(partition_values):
                pruned_files.append(file)
        
        if not pruned_files:
            logger.warning(f"Partition pruning removed all partitions, using all {len(all_files)} files")
            pruned_files = all_files
        elif len(pruned_files) < len(all_files):
            logger.info(f"Partition pruning: {len(pruned_files)}/{len(all_files)} partitions remaining")
        
        return self._validate_partitions(pruned_files)
    
    def _partition_matches_filters(self, partition_values: Dict[str, Any]) -> bool:
        """Check if partition values match filters (converted from query)."""
        for col, op, val in self._filters:
            if col not in partition_values:
                continue  # Filter on non-partition column
            
            pval = partition_values[col]
            
            # Check condition
            if op == '==' and pval != val:
                return False
            elif op == '!=' and pval == val:
                return False
            elif op == '<' and pval >= val:
                return False
            elif op == '<=' and pval > val:
                return False
            elif op == '>' and pval <= val:
                return False
            elif op == '>=' and pval < val:
                return False
            elif op == 'in' and pval not in val:
                return False
            elif op == 'not in' and pval in val:
                return False
        
        return True
    
    def _validate_partitions(self, files: List[Path]) -> List[Path]:
        """Validate partition files."""
        valid_partitions = []
        for file in files:
            try:
                if file.stat().st_size < 1024:
                    logger.debug(f"Skipping small partition: {file.name}")
                    continue
                valid_partitions.append(file)
            except Exception as e:
                logger.warning(f"Cannot access partition {file}: {e}")
        
        if not valid_partitions:
            raise ValueError(f"No valid partitions found")
        
        return sorted(valid_partitions)
    
    def _open_parquet_file(self, path: Path):
        """Open parquet file with current backend."""
        if self._parquet_backend == 'fastparquet':
            return fp.ParquetFile(path)
        else:
            return pq.ParquetFile(path)
    
    def _get_parquet_row_count(self, parquet_file) -> int:
        """Get row count from parquet file."""
        if self._parquet_backend == 'fastparquet':
            return parquet_file.count()
        else:
            return parquet_file.metadata.num_rows
    
    def _read_sample_parquet(self, parquet_file, n_rows: int) -> pd.DataFrame:
        """Read sample from parquet file."""
        if self._parquet_backend == 'fastparquet':
            # Fastparquet doesn't have a direct row limit parameter
            # Read first row group and take first n_rows
            df = parquet_file.to_pandas()
            return df.head(n_rows)
        else:
            batch = next(parquet_file.iter_batches(batch_size=n_rows))
            return batch.to_pandas()
    
    def iter_chunks(self, columns: Optional[List[str]] = None):
        """
        Iterate over data in chunks with efficient filtering and projection.
        
        Parameters:
        -----------
        columns : list of str, optional
            Columns to load (projection pushdown for Parquet). If None, loads all columns.
        
        Yields:
        -------
        DataFrame chunks (filtered by query/filters)
        """
        # Determine columns to load (including filter columns for Parquet)
        load_columns = self._get_load_columns(columns)
        
        if self.info.source_type == 'dataframe':
            # For DataFrame, data is already filtered - just chunk it
            for i in range(0, self.info.n_rows, self.chunk_size):
                chunk = self._dataframe.iloc[i:i+self.chunk_size]
                if columns:
                    chunk = chunk[columns]
                yield chunk
        
        elif self.info.source_type == 'parquet':
            # Use backend-specific reading with filter pushdown
            yield from self._iter_parquet_chunks(self._parquet_file, load_columns, columns)
        
        elif self.info.source_type == 'partitioned':
            # Iterate through partitions
            for partition_file in self.info.partitions:
                try:
                    parquet_file = self._open_parquet_file(partition_file)
                    yield from self._iter_parquet_chunks(parquet_file, load_columns, columns)
                except Exception as e:
                    logger.warning(f"Failed to read partition {partition_file.name}: {e}")
                    continue
    
    def _get_load_columns(self, requested_columns: Optional[List[str]]) -> Optional[List[str]]:
        """Determine which columns to load (includes filter columns)."""
        if requested_columns is None:
            return None
        
        # For Parquet: load requested columns + filter columns
        if self.info.source_type in ['parquet', 'partitioned']:
            filter_cols = self._extract_query_columns()
            return list(set(requested_columns) | set(filter_cols))
        
        return requested_columns
    
    def _iter_parquet_chunks(self, parquet_file, load_columns: Optional[List[str]], 
                            final_columns: Optional[List[str]]):
        """Iterate chunks from parquet file with filtering."""
        if self._parquet_backend == 'fastparquet':
            # Use fastparquet with filter pushdown if filters were converted from query
            if self._filters:
                # Fastparquet expects filters as list of tuples for AND operations
                try:
                    # Apply filters at read time
                    df = parquet_file.to_pandas(columns=load_columns, filters=self._filters)
                    # Chunk the filtered result
                    for i in range(0, len(df), self.chunk_size):
                        chunk = df.iloc[i:i+self.chunk_size]
                        if final_columns:
                            chunk = chunk[final_columns]
                        yield chunk
                except Exception as e:
                    logger.warning(f"Filter pushdown failed, applying query after read: {e}")
                    # Fallback: read all and apply query
                    df = parquet_file.to_pandas(columns=load_columns)
                    if self.query:
                        df = df.query(self.query)
                    for i in range(0, len(df), self.chunk_size):
                        chunk = df.iloc[i:i+self.chunk_size]
                        if final_columns:
                            chunk = chunk[final_columns]
                        yield chunk
            else:
                # No filters, stream row groups directly
                for df in parquet_file.iter_row_groups(columns=load_columns):
                    # Apply query if it couldn't be converted to filters
                    if self.query:
                        df = df.query(self.query)
                    if len(df) > 0:
                        # Project to final columns after filtering
                        if final_columns:
                            df = df[final_columns]
                        for i in range(0, len(df), self.chunk_size):
                            yield df.iloc[i:i+self.chunk_size]
        else:
            # Use pyarrow - no native filter pushdown, apply query after read
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size, columns=load_columns):
                chunk = batch.to_pandas()
                
                # Apply query if provided
                if self.query:
                    chunk = chunk.query(self.query)
                
                if len(chunk) > 0:
                    # Project to final columns after filtering
                    if final_columns:
                        chunk = chunk[final_columns]
                    yield chunk
    
    def iter_chunks_parallel(self, columns: Optional[List[str]] = None, 
                            n_workers: Optional[int] = None) -> Iterator[Tuple[int, pd.DataFrame]]:
        """
        Iterate over chunks in parallel with efficient filtering.
        
        Parameters:
        -----------
        columns : list of str, optional
            Columns to load (projection pushdown)
        n_workers : int, optional
            Number of parallel workers
        
        Yields:
        -------
        tuple: (chunk_id, chunk_df) (filtered and projected)
        """
        chunk_id = 0
        for chunk in self.iter_chunks(columns=columns):
            yield (chunk_id, chunk)
            chunk_id += 1
    
    def estimate_n_chunks(self) -> int:
        """Estimate total number of chunks."""
        return max(1, self.info.n_rows // self.chunk_size)
    
    def supports_parallel(self) -> bool:
        """Check if data source supports efficient parallel processing."""
        return True
    
    def validate_columns(self, required_cols: List[str]) -> None:
        """Validate that required columns exist."""
        missing = [col for col in required_cols if col not in self.info.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def get_numeric_columns(self, exclude: Optional[List[str]] = None) -> List[str]:
        """Get list of numeric columns, optionally excluding some."""
        exclude = exclude or []
        return [col for col in self.info.numeric_columns if col not in exclude]
    
    def get_schema_sample(self) -> pd.DataFrame:
        """Get a small sample for schema validation."""
        if self.info.source_type == 'dataframe':
            return self._dataframe.head(100)
        elif self.info.source_type == 'parquet':
            return self._read_sample_parquet(self._parquet_file, 100)
        elif self.info.source_type == 'partitioned':
            first_file = self._open_parquet_file(self.info.partitions[0])
            return self._read_sample_parquet(first_file, 100)
