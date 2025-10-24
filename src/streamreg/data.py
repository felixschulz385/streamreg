import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Iterator, Tuple
from dataclasses import dataclass
import logging
import fastparquet as fp
import re
import pyarrow.parquet as pq

# Conditional imports for pyarrow.dataset
try:
    import pyarrow.dataset as ds
    import pyarrow as pa
    PYARROW_DATASET_AVAILABLE = True
except ImportError:
    ds = None
    pa = None
    PYARROW_DATASET_AVAILABLE = False

logger = logging.getLogger(__name__)


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


@dataclass
class ChunkTask:
    chunk_id: int
    file_path: str
    row_groups: Optional[List[int]]
    row_start: Optional[int] = None
    row_end: Optional[int] = None
    columns: Optional[List[str]] = None
    filters: Optional[List[Tuple]] = None
    query: Optional[str] = None
    partition_idx: Optional[int] = None
    feature_engineering: Optional[Dict] = None
    add_intercept: bool = False


class StreamData:
    """
    Unified data interface for streaming regression with efficient subsetting.
    
    Supports:
    - Pandas DataFrame (in-memory, filtered once at initialization)
    - Single parquet file (with filter pushdown)
    - Partitioned parquet dataset (with partition pruning and filter pushdown)
    
    All parallel processing is handled internally - users don't need to know about partitioning.
    
    Performance optimizations:
    - For DataFrames: Query applied once at initialization, then chunked
    - For Parquet: Query converted to filters for pushdown when possible, otherwise applied after read
    - For Partitioned Parquet: Partition pruning before reading + PyArrow dataset scanner for efficient I/O
    - Only loads columns needed for modeling, drops filter-only columns after filtering
    """
    
    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        chunk_size: int = 10000,
        query: Optional[str] = None
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
        """
        self.chunk_size = chunk_size
        self.query = query
        
        # Internal: converted filters for Parquet pushdown
        self._filters = None
        
        # Compiled filter objects for reuse
        self._compiled_query = None
        self._filter_columns = set()
        
        self._setup_data_source(data)
        
        # Validate and compile query if provided
        if self.query:
            self._validate_and_compile_filters()
    
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
        if not query:
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
        self._parquet_file = fp.ParquetFile(path)
        schema = self._parquet_file.columns
        columns = list(schema)
        
        # Read small sample to determine numeric columns
        sample_df = self._read_sample_parquet(self._parquet_file, 100)
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_rows = self._parquet_file.count()
        
        self.info = DatasetInfo(
            n_rows=n_rows,
            n_cols=len(columns),
            columns=columns,
            numeric_columns=numeric_cols,
            source_type='parquet',
            source_path=path
        )
        self._dataframe = None
        
        logger.debug(f"Loaded parquet: {self.info.n_rows:,} rows, {self.info.n_cols} columns")
    
    def _setup_partitioned_parquet(self, path: Path):
        """Setup from partitioned parquet dataset with partition pruning."""
        partitions = self._discover_and_prune_partitions(path)
        
        # Read first partition for schema
        first_parquet = fp.ParquetFile(partitions[0])
        columns = list(first_parquet.columns)
        
        # Estimate total rows from remaining partitions
        total_rows = sum(
            fp.ParquetFile(p).count()
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
        
        logger.info(f"Loaded partitioned dataset: {total_rows:,} rows, {len(partitions)} partitions")
    
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
    
    def _read_sample_parquet(self, parquet_file, n_rows: int) -> pd.DataFrame:
        """Read sample from parquet file."""
        try:
            # Use PyArrow for efficient row group reading
            # Convert fastparquet file path to pyarrow ParquetFile if needed
            if hasattr(parquet_file, 'fn'):
                # It's a fastparquet file, get the path
                file_path = parquet_file.fn
                pq_file = pq.ParquetFile(file_path)
            else:
                # Assume it's already a path-like object
                pq_file = pq.ParquetFile(str(parquet_file))
            
            # Read first row group using PyArrow (more efficient)
            table = pq_file.read_row_group(0, use_pandas_metadata=False)
            df = table.to_pandas()
            return df.head(n_rows)
        except (AttributeError, IndexError, Exception) as e:
            # Fallback: use fastparquet if available
            logger.debug(f"PyArrow read_row_group failed ({e}), falling back to fastparquet")
            try:
                if hasattr(parquet_file, 'to_pandas'):
                    # It's a fastparquet file
                    df = parquet_file.to_pandas(row_groups=[0])
                else:
                    # Try to create fastparquet file from path
                    fp_file = fp.ParquetFile(str(parquet_file))
                    df = fp_file.to_pandas(row_groups=[0])
                return df.head(n_rows)
            except Exception as e2:
                logger.warning(f"Both PyArrow and fastparquet sampling failed: {e2}")
                # Last resort: read entire file (inefficient but works)
                if hasattr(parquet_file, 'to_pandas'):
                    df = parquet_file.to_pandas()
                else:
                    fp_file = fp.ParquetFile(str(parquet_file))
                    df = fp_file.to_pandas()
                return df.head(n_rows)
    
    def _filters_to_pyarrow_expression(self, filters: List[Tuple]) -> Optional['pa.compute.Expression']:
        """
        Convert simple filters to PyArrow dataset filter expression.
        
        Returns None if conversion fails (for fallback to pandas filtering).
        """
        if not PYARROW_DATASET_AVAILABLE or not filters:
            return None
        
        try:
            import pyarrow.compute as pc
            
            expressions = []
            for col, op, val in filters:
                field = ds.field(col)
                
                if op == '==':
                    expr = field == val
                elif op == '!=':
                    expr = field != val
                elif op == '<':
                    expr = field < val
                elif op == '<=':
                    expr = field <= val
                elif op == '>':
                    expr = field > val
                elif op == '>=':
                    expr = field >= val
                elif op == 'in':
                    expr = field.isin(val)
                elif op == 'not in':
                    expr = ~field.isin(val)
                else:
                    logger.debug(f"Unsupported operator for PyArrow filter: {op}")
                    return None
                
                expressions.append(expr)
            
            # Combine with AND
            if len(expressions) == 1:
                return expressions[0]
            else:
                combined = expressions[0]
                for expr in expressions[1:]:
                    combined = combined & expr
                return combined
        
        except Exception as e:
            logger.debug(f"Failed to convert filters to PyArrow expression: {e}")
            return None
    
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
        
        Notes
        -----
        For worker-side parquet reads use `create_chunk_tasks()` / `iter_tasks()`.
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
            # Use fastparquet with filter pushdown
            yield from self._iter_parquet_chunks(self._parquet_file, load_columns, columns)
        
        elif self.info.source_type == 'partitioned':
            # Use PyArrow dataset scanner if available for better performance
            if PYARROW_DATASET_AVAILABLE:
                try:
                    yield from self._iter_partitioned_with_dataset(load_columns, columns)
                    return  # Success, exit early
                except Exception as e:
                    logger.warning(f"PyArrow dataset reading failed ({e}), falling back to per-file reading")
            
            # Fallback: Iterate through partitions with per-file reading
            for partition_file in self.info.partitions:
                try:
                    parquet_file = fp.ParquetFile(partition_file)
                    yield from self._iter_parquet_chunks(parquet_file, load_columns, columns)
                except Exception as e:
                    logger.warning(f"Failed to read partition {partition_file.name}: {e}")
                    continue
    
    def _iter_partitioned_with_dataset(self, load_columns: Optional[List[str]], 
                                       final_columns: Optional[List[str]]):
        """
        Iterate chunks from partitioned dataset using PyArrow dataset scanner.
        
        This provides better performance through:
        - Parallel reading across partitions
        - Efficient filter pushdown
        - Optimized columnar I/O
        """
        # Build dataset from source path
        dataset = ds.dataset(
            str(self.info.source_path),
            format="parquet",
            partitioning="hive"  # Support Hive-style partitioning
        )
        
        # Convert filters to PyArrow expression
        pyarrow_filter = None
        query_applied = False
        
        if self._filters:
            pyarrow_filter = self._filters_to_pyarrow_expression(self._filters)
            if pyarrow_filter is not None:
                query_applied = True
                logger.debug("Using PyArrow filter pushdown for partitioned dataset")
            else:
                logger.debug("Cannot convert filters to PyArrow, will apply query after read")
        
        # Create scanner with filter pushdown
        scanner = dataset.scanner(
            columns=load_columns,
            filter=pyarrow_filter,
            use_threads=True
        )
        
        # Iterate through record batches
        for record_batch in scanner.to_batches():
            # Convert to pandas
            df = record_batch.to_pandas()
            
            # Apply query if it wasn't pushed down
            if self.query and not query_applied:
                df = df.query(self.query)
            
            if len(df) == 0:
                continue
            
            # Project to final columns after filtering
            if final_columns:
                df = df[final_columns]
            
            # Chunk the batch if it's larger than chunk_size
            for i in range(0, len(df), self.chunk_size):
                chunk = df.iloc[i:i+self.chunk_size]
                if len(chunk) > 0:
                    yield chunk
    
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
        # Use PyArrow for efficient row group reading
        try:
            # Get file path from fastparquet file
            if hasattr(parquet_file, 'fn'):
                file_path = parquet_file.fn
            else:
                file_path = str(parquet_file)
            
            pq_file = pq.ParquetFile(file_path)
            
            # Iterate through row groups using PyArrow
            for rg_idx in range(pq_file.num_row_groups):
                try:
                    # Read row group efficiently with PyArrow
                    table = pq_file.read_row_group(
                        rg_idx,
                        columns=load_columns,
                        use_threads=True,
                        use_pandas_metadata=False
                    )
                    df = table.to_pandas()
                    
                    # Apply query if it couldn't be converted to filters
                    if self.query and not self._filters:
                        df = df.query(self.query)
                    
                    if len(df) > 0:
                        # Project to final columns after filtering
                        if final_columns:
                            df = df[final_columns]
                        
                        # Chunk the row group if it's larger than chunk_size
                        for i in range(0, len(df), self.chunk_size):
                            yield df.iloc[i:i+self.chunk_size]
                            
                except Exception as e:
                    logger.warning(f"Failed to read row group {rg_idx} with PyArrow: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"PyArrow iteration failed ({e}), falling back to fastparquet")
            # Fallback to fastparquet implementation
            if self._filters:
                try:
                    df = parquet_file.to_pandas(columns=load_columns, filters=self._filters)
                    for i in range(0, len(df), self.chunk_size):
                        chunk = df.iloc[i:i+self.chunk_size]
                        if final_columns:
                            chunk = chunk[final_columns]
                        yield chunk
                except Exception as e:
                    logger.warning(f"Filter pushdown failed, applying query after read: {e}")
                    df = parquet_file.to_pandas(columns=load_columns)
                    if self.query:
                        df = df.query(self.query)
                    for i in range(0, len(df), self.chunk_size):
                        chunk = df.iloc[i:i+self.chunk_size]
                        if final_columns:
                            chunk = chunk[final_columns]
                        yield chunk
            else:
                for df in parquet_file.iter_row_groups(columns=load_columns):
                    if self.query:
                        df = df.query(self.query)
                    if len(df) > 0:
                        if final_columns:
                            df = df[final_columns]
                        for i in range(0, len(df), self.chunk_size):
                            yield df.iloc[i:i+self.chunk_size]
    
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
        
        Notes
        -----
        For worker-side parquet reads use `create_chunk_tasks()` / `iter_tasks()`.
        """
        chunk_id = 0
        for chunk in self.iter_chunks(columns=columns):
            yield (chunk_id, chunk)
            chunk_id += 1
    
    def supports_parallel(self) -> bool:
        """Check if data source supports efficient parallel processing.

        This includes worker-side parquet reads via `create_chunk_tasks()` / `iter_tasks()`.
        """
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
            first_file = fp.ParquetFile(self.info.partitions[0])
            return self._read_sample_parquet(first_file, 100)
    
    def create_chunk_tasks(self, requested_columns: Optional[List[str]], 
                           target_chunk_rows: Optional[int] = None, 
                           bundle_rowgroups: bool = True) -> List[ChunkTask]:
        if self.info.source_type == 'dataframe':
            raise ValueError("Chunk task creation is only supported for parquet-backed sources")
        target_rows = target_chunk_rows or self.chunk_size
        target_rows = max(1, int(target_rows))
        columns_to_load = self._get_load_columns(requested_columns)
        columns_list = list(columns_to_load) if columns_to_load is not None else None
        filter_list = [tuple(f) for f in self._filters] if self._filters else None

        if self.info.source_type == 'parquet':
            source_files = [(0, self.info.source_path)]
        else:
            source_files = list(enumerate(self.info.partitions or []))

        tasks: List[ChunkTask] = []

        for partition_idx, path in source_files:
            if path is None:
                continue
            path_obj = Path(path)
            try:
                path_obj.stat()
            except Exception as exc:
                logger.warning(f"Skipping parquet file {path_obj}: {exc}")
                continue
            try:
                pq_file = pq.ParquetFile(str(path_obj))
            except Exception as exc:
                logger.warning(f"Skipping parquet file {path_obj}: {exc}")
                continue

            column_names = list(pq_file.schema.names)
            if columns_list:
                missing = [col for col in columns_list if col not in column_names]
                if missing:
                    logger.warning(f"Skipping parquet file {path_obj}, missing columns: {missing}")
                    continue

            row_group_sizes = self._parquet_row_group_sizes(pq_file)
            file_key = str(path_obj)

            def add_task(row_groups: Optional[List[int]], row_start: Optional[int], row_end: Optional[int]):
                if row_start is not None and row_end is not None and row_start >= row_end:
                    return
                tasks.append(ChunkTask(
                    chunk_id=-1,
                    file_path=file_key,
                    row_groups=row_groups.copy() if row_groups is not None else None,
                    row_start=row_start,
                    row_end=row_end,
                    columns=columns_list.copy() if columns_list is not None else None,
                    filters=filter_list.copy() if filter_list is not None else None,
                    query=self.query,
                    partition_idx=partition_idx,
                ))

            if row_group_sizes:
                pending_groups: List[int] = []
                pending_rows = 0
                for rg_idx, rg_rows in enumerate(row_group_sizes):
                    if rg_rows <= 0:
                        continue
                    if rg_rows >= target_rows:
                        if pending_groups:
                            add_task(pending_groups, None, None)
                            pending_groups = []
                            pending_rows = 0
                        offset = 0
                        while offset < rg_rows:
                            end = min(rg_rows, offset + target_rows)
                            add_task([rg_idx], offset, end)
                            offset = end
                        continue

                    pending_groups.append(rg_idx)
                    pending_rows += rg_rows
                    if not bundle_rowgroups or pending_rows >= target_rows:
                        add_task(pending_groups, None, None)
                        pending_groups = []
                        pending_rows = 0

                if pending_groups:
                    add_task(pending_groups, None, None)
            else:
                total_rows = int(pq_file.metadata.num_rows or 0)
                if total_rows <= 0:
                    continue
                offset = 0
                while offset < total_rows:
                    end = min(total_rows, offset + target_rows)
                    add_task(None, offset, end)
                    offset = end

        return tasks

    def _parquet_row_group_sizes(self, pq_file) -> List[int]:
        metadata = getattr(pq_file, "metadata", None)
        if metadata is None:
            return []
        return [
            int(metadata.row_group(i).num_rows or 0)
            for i in range(metadata.num_row_groups)
        ]
    
    def iter_tasks(self, requested_columns: Optional[List[str]], 
                   target_chunk_rows: Optional[int] = None,
                   bundle_rowgroups: bool = True) -> Iterator[Tuple[int, ChunkTask]]:
        """
        Create and yield chunk tasks for worker-side parquet reads.
        
        Parameters:
        -----------
        requested_columns : list of str, optional
            Columns to load
        target_chunk_rows : int, optional
            Target rows per chunk (defaults to self.chunk_size)
        bundle_rowgroups : bool
            Whether to bundle multiple row groups into single tasks
        
        Yields:
        -------
        tuple: (chunk_id, ChunkTask)
        """
        tasks = self.create_chunk_tasks(requested_columns, target_chunk_rows, bundle_rowgroups)
        # Assign chunk IDs
        for idx, task in enumerate(tasks):
            task.chunk_id = idx
            yield idx, task
