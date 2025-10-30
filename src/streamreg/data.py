import numpy as np
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Iterator, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Metadata about a dataset."""
    n_rows: int
    n_cols: int
    columns: List[str]
    numeric_columns: List[str]
    source_type: str  # 'dataframe', 'parquet'
    source_path: Optional[Path] = None


class StreamData:
    """
    Unified data interface for streaming regression with efficient subsetting.
    
    Uses Dask DataFrame with PyArrow backend for both in-memory and out-of-core processing.
    
    Supports:
    - Pandas DataFrame (converted to Dask DataFrame)
    - Single or partitioned parquet dataset (loaded as Dask DataFrame)
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
            Size of chunks for iteration (used as Dask partition size)
        query : str, optional
            Pandas query string to filter data
        """
        self.chunk_size = chunk_size
        self.query = query
        self._dask_df = None
        self._transform_func = None  # Store optional transformation function
        
        self._setup_data_source(data)
        
        # Apply query filter if provided
        if self.query:
            self._apply_query_filter()
    
    def _apply_query_filter(self):
        """Apply query filter to Dask DataFrame."""
        if self.query and self._dask_df is not None:
            try:
                self._dask_df = self._dask_df.query(self.query)
                logger.info(f"Query filter applied: {self.query}")
            except Exception as e:
                raise ValueError(f"Invalid query string '{self.query}': {e}")
    
    def _setup_data_source(self, data: Union[str, Path, pd.DataFrame]):
        """Setup data source and extract metadata."""
        if isinstance(data, pd.DataFrame):
            self._setup_dataframe(data)
        else:
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Data source not found: {path}")
            
            if path.is_dir() or path.suffix == '.parquet':
                self._setup_parquet(path)
            else:
                raise ValueError(f"Unsupported data source: {path}")
    
    def _setup_dataframe(self, df: pd.DataFrame):
        """Setup from DataFrame by converting to Dask DataFrame."""
        # Compute total memory usage
        total_memory = df.memory_usage(deep=True).sum()
        target_memory = 100 * 1024 * 1024  # 100MB in bytes
        ideal_npartitions = max(1, total_memory // target_memory)
        
        # Convert to Dask DataFrame with calculated partitions
        self._dask_df = dd.from_pandas(df, npartitions=ideal_npartitions)
        
        # Get metadata from pandas DataFrame
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.info = DatasetInfo(
            n_rows=len(df),
            n_cols=len(df.columns),
            columns=df.columns.tolist(),
            numeric_columns=numeric_cols,
            source_type='dataframe'
        )
        
        logger.debug(f"Loaded DataFrame as Dask: {self.info.n_rows:,} rows, {self.info.n_cols} columns, {ideal_npartitions} partitions (target ~100MB each)")
    
    def _setup_parquet(self, path: Path):
        """Setup from parquet file(s) as Dask DataFrame."""
        # Dask automatically handles single files and partitioned datasets
        # Use PyArrow engine for better performance
        try:
            self._dask_df = dd.read_parquet(
                str(path),
                engine='pyarrow',
                calculate_divisions=False  # Faster loading, can compute later if needed
            )
            
            # Get a sample to estimate memory usage
            sample = self._dask_df.head(1000, npartitions=1)
            df_sample = pd.DataFrame(sample)
            sample_memory = df_sample.memory_usage(deep=True).sum() if len(df_sample) > 0 else 0
            avg_row_size = sample_memory / len(df_sample) if len(df_sample) > 0 else 0
            
            # Estimate total rows and memory
            estimated_rows = len(self._dask_df)  # Triggers compute
            total_memory = avg_row_size * estimated_rows
            target_memory = 100 * 1024 * 1024  # 100MB in bytes
            ideal_npartitions = max(1, int(total_memory // target_memory))
            
            # Repartition to target memory size
            self._dask_df = self._dask_df.repartition(npartitions=ideal_npartitions)
            logger.debug(f"Repartitioned to {ideal_npartitions} partitions (target ~100MB each)")
            
            # Get metadata (lightweight operations)
            columns = self._dask_df.columns.tolist()
            
            # Determine numeric columns from dtypes
            numeric_cols = [
                col for col, dtype in self._dask_df.dtypes.items()
                if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_categorical_dtype(dtype)
            ]
            
            # Get row count (this triggers computation)
            n_rows = len(self._dask_df)
            
            self.info = DatasetInfo(
                n_rows=n_rows,
                n_cols=len(columns),
                columns=columns,
                numeric_columns=numeric_cols,
                source_type='parquet',
                source_path=path
            )
            
            logger.info(f"Loaded parquet as Dask: {self.info.n_rows:,} rows, {self.info.n_cols} columns, {self._dask_df.npartitions} partitions (target ~100MB each)")
        
        except Exception as e:
            raise ValueError(f"Failed to load parquet from {path}: {e}")
    
    def iter_chunks(self, columns: Optional[List[str]] = None):
        """
        Iterate over data in chunks with efficient filtering and projection.
        
        Parameters:
        -----------
        columns : list of str, optional
            Columns to load (projection). If None, loads all columns.
        
        Yields:
        -------
        DataFrame chunks (filtered by query if provided)
        """
        # Apply column projection
        df_to_iterate = self._dask_df
        if columns is not None:
            # Validate columns
            missing = [col for col in columns if col not in self.info.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            df_to_iterate = df_to_iterate[columns]
        
        # Iterate over Dask partitions
        for partition in df_to_iterate.to_delayed():
            # Compute each partition (converts to pandas DataFrame)
            chunk = partition.compute()
            
            # Yield non-empty chunks
            if len(chunk) > 0:
                yield chunk
    
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
        return self._dask_df.head(100, npartitions=1)
    
    def with_transform(self, transform_func):
        """
        Apply a lazy transformation to the data pipeline.
        
        Parameters:
        -----------
        transform_func : callable
            Function that takes a DataFrame (or Dask DataFrame) and returns a transformed 
            DataFrame. Should return a Dask DataFrame for lazy evaluation.
        
        Returns:
        --------
        StreamData : A new StreamData instance with the transformation applied
        """
        # Create a shallow copy
        import copy
        new_instance = copy.copy(self)
        
        # Apply transformation to the Dask DataFrame
        try:
            transformed_ddf = transform_func(self._dask_df)
            
            # Ensure result is a Dask DataFrame
            if isinstance(transformed_ddf, pd.DataFrame):
                transformed_ddf = dd.from_pandas(transformed_ddf, npartitions=self._dask_df.npartitions)
            
            new_instance._dask_df = transformed_ddf
            new_instance._transform_func = transform_func
            
            logger.info("Transformation applied to data pipeline (lazy)")
            
        except Exception as e:
            raise ValueError(f"Failed to apply transformation: {e}")
        
        return new_instance
