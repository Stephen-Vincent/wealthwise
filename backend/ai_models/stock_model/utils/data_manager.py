"""
Data Management Utilities

This module provides utilities for managing cached market data, analysis results,
and other data files used by the WealthWise recommendation system.

Key Features:
1. Market data caching and management
2. Analysis result storage and retrieval
3. Automatic data cleanup and maintenance
4. Performance monitoring and optimization
5. Data integrity and validation
"""

import os
import json
import pandas as pd
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import hashlib
import shutil

logger = logging.getLogger(__name__)


class DataManager:
    """
    Centralized data management system for all cached data
    
    Handles caching, storage, retrieval, and maintenance of market data,
    analysis results, and other system data.
    """
    
    def __init__(self, data_dir: str = "./ai_models/stock_model/data"):
        """
        Initialize data manager
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = data_dir
        
        # Create directory structure
        self.subdirs = {
            "market_data": os.path.join(data_dir, "market_data"),
            "analysis": os.path.join(data_dir, "analysis"), 
            "cache": os.path.join(data_dir, "cache"),
            "sessions": os.path.join(data_dir, "sessions"),
            "correlations": os.path.join(data_dir, "correlations"),
            "portfolios": os.path.join(data_dir, "portfolios")
        }
        
        for subdir in self.subdirs.values():
            os.makedirs(subdir, exist_ok=True)
        
        # Cache settings
        self.cache_settings = {
            "market_data_ttl": timedelta(hours=24),      # Refresh daily
            "analysis_ttl": timedelta(hours=6),          # Refresh every 6 hours
            "correlation_ttl": timedelta(days=7),        # Refresh weekly
            "session_ttl": timedelta(hours=24),          # Clean after 24 hours
            "max_cache_size_mb": 500                     # Maximum cache size
        }
        
        self.metadata_file = os.path.join(data_dir, "data_metadata.json")
        self.load_metadata()
    
    def get_cache_path(self, category: str, identifier: str, extension: str = ".json") -> str:
        """
        Get path for cached data file
        
        Args:
            category: Data category (market_data, analysis, etc.)
            identifier: Unique identifier for the data
            extension: File extension
            
        Returns:
            Full path to cache file
        """
        if category not in self.subdirs:
            raise ValueError(f"Unknown data category: {category}")
        
        filename = f"{identifier}{extension}"
        return os.path.join(self.subdirs[category], filename)
    
    def cache_market_data(self, ticker: str, period: str, data: pd.DataFrame) -> str:
        """
        Cache market data for a stock/ETF
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1y, 2y, etc.)
            data: Market data DataFrame
            
        Returns:
            Path to cached file
        """
        try:
            # Create identifier
            date_str = datetime.now().strftime("%Y-%m-%d")
            identifier = f"{ticker}_{period}_{date_str}"
            
            # Save data
            cache_path = self.get_cache_path("market_data", identifier, ".csv")
            data.to_csv(cache_path)
            
            # Update metadata
            self._update_cache_metadata("market_data", identifier, {
                "ticker": ticker,
                "period": period,
                "rows": len(data),
                "columns": list(data.columns),
                "date_range": {
                    "start": str(data.index.min()) if not data.empty else None,
                    "end": str(data.index.max()) if not data.empty else None
                },
                "file_size_mb": os.path.getsize(cache_path) / (1024*1024)
            })
            
            logger.debug(f"📊 Cached market data: {ticker} ({len(data)} rows)")
            return cache_path
            
        except Exception as e:
            logger.error(f"Failed to cache market data for {ticker}: {e}")
            return ""
    
    def get_cached_market_data(self, ticker: str, period: str, 
                              max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Retrieve cached market data if available and fresh
        
        Args:
            ticker: Stock ticker symbol
            period: Time period
            max_age_hours: Maximum age of cached data in hours
            
        Returns:
            DataFrame with market data or None if not available/stale
        """
        try:
            # Find most recent cache file for this ticker/period
            cache_files = []
            for filename in os.listdir(self.subdirs["market_data"]):
                if filename.startswith(f"{ticker}_{period}_") and filename.endswith(".csv"):
                    filepath = os.path.join(self.subdirs["market_data"], filename)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    cache_files.append((filepath, mod_time))
            
            if not cache_files:
                return None
            
            # Get most recent file
            latest_file, latest_time = max(cache_files, key=lambda x: x[1])
            
            # Check if fresh enough
            age = datetime.now() - latest_time
            if age > timedelta(hours=max_age_hours):
                logger.debug(f"Cache for {ticker}_{period} is stale ({age.total_seconds()/3600:.1f}h old)")
                return None
            
            # Load and return data
            data = pd.read_csv(latest_file, index_col=0, parse_dates=True)
            logger.debug(f"📊 Retrieved cached data: {ticker} ({len(data)} rows)")
            return data
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached data for {ticker}: {e}")
            return None
    
    def cache_analysis_result(self, analysis_type: str, identifier: str, 
                            result: Any, metadata: Dict[str, Any] = None) -> str:
        """
        Cache analysis results
        
        Args:
            analysis_type: Type of analysis (factor_analysis, correlation, etc.)
            identifier: Unique identifier for this analysis
            result: Analysis result to cache
            metadata: Additional metadata about the analysis
            
        Returns:
            Path to cached file
        """
        try:
            # Create cache identifier
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_id = f"{analysis_type}_{identifier}_{timestamp}"
            
            # Determine file format based on result type
            if isinstance(result, pd.DataFrame):
                cache_path = self.get_cache_path("analysis", cache_id, ".csv")
                result.to_csv(cache_path)
            elif isinstance(result, dict) or isinstance(result, list):
                cache_path = self.get_cache_path("analysis", cache_id, ".json")
                with open(cache_path, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
            else:
                # Use pickle for other types
                cache_path = self.get_cache_path("analysis", cache_id, ".pickle")
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
            
            # Update metadata
            cache_metadata = {
                "analysis_type": analysis_type,
                "identifier": identifier,
                "result_type": type(result).__name__,
                "file_size_mb": os.path.getsize(cache_path) / (1024*1024)
            }
            if metadata:
                cache_metadata.update(metadata)
            
            self._update_cache_metadata("analysis", cache_id, cache_metadata)
            
            logger.debug(f"💾 Cached analysis result: {analysis_type}_{identifier}")
            return cache_path
            
        except Exception as e:
            logger.error(f"Failed to cache analysis result: {e}")
            return ""
    
    def get_cached_analysis(self, analysis_type: str, identifier: str,
                          max_age_hours: int = 6) -> Optional[Any]:
        """
        Retrieve cached analysis result
        
        Args:
            analysis_type: Type of analysis
            identifier: Analysis identifier
            max_age_hours: Maximum age in hours
            
        Returns:
            Analysis result or None if not available/stale
        """
        try:
            # Find matching cache files
            cache_files = []
            pattern = f"{analysis_type}_{identifier}_"
            
            for filename in os.listdir(self.subdirs["analysis"]):
                if filename.startswith(pattern):
                    filepath = os.path.join(self.subdirs["analysis"], filename)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    cache_files.append((filepath, mod_time))
            
            if not cache_files:
                return None
            
            # Get most recent file
            latest_file, latest_time = max(cache_files, key=lambda x: x[1])
            
            # Check freshness
            age = datetime.now() - latest_time
            if age > timedelta(hours=max_age_hours):
                return None
            
            # Load based on file extension
            if latest_file.endswith('.csv'):
                result = pd.read_csv(latest_file, index_col=0)
            elif latest_file.endswith('.json'):
                with open(latest_file, 'r') as f:
                    result = json.load(f)
            elif latest_file.endswith('.pickle'):
                with open(latest_file, 'rb') as f:
                    result = pickle.load(f)
            else:
                return None
            
            logger.debug(f"💾 Retrieved cached analysis: {analysis_type}_{identifier}")
            return result
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached analysis: {e}")
            return None
    
    def cleanup_old_data(self, max_age_days: int = 7) -> Dict[str, Any]:
        """
        Clean up old cached data files
        
        Args:
            max_age_days: Remove files older than this many days
            
        Returns:
            Dict with cleanup results
        """
        logger.info(f"🧹 Cleaning up data older than {max_age_days} days...")
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cleanup_results = {
            "removed_files": [],
            "errors": [],
            "space_freed_mb": 0,
            "files_processed": 0
        }
        
        for category, directory in self.subdirs.items():
            try:
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    
                    if os.path.isfile(filepath):
                        file_mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        cleanup_results["files_processed"] += 1
                        
                        if file_mod_time < cutoff_date:
                            try:
                                file_size = os.path.getsize(filepath) / (1024*1024)  # MB
                                os.remove(filepath)
                                
                                cleanup_results["removed_files"].append({
                                    "category": category,
                                    "filename": filename,
                                    "age_days": (datetime.now() - file_mod_time).days,
                                    "size_mb": file_size
                                })
                                cleanup_results["space_freed_mb"] += file_size
                                
                            except Exception as e:
                                cleanup_results["errors"].append(f"Failed to remove {filepath}: {e}")
                                
            except Exception as e:
                cleanup_results["errors"].append(f"Error processing {category}: {e}")
        
        logger.info(f"✅ Cleanup complete: removed {len(cleanup_results['removed_files'])} files, "
                   f"freed {cleanup_results['space_freed_mb']:.1f}MB")
        
        return cleanup_results
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics
        
        Returns:
            Dict with cache statistics
        """
        stats = {
            "categories": {},
            "total_files": 0,
            "total_size_mb": 0,
            "oldest_file": None,
            "newest_file": None,
            "cache_health": "good"
        }
        
        oldest_time = datetime.now()
        newest_time = datetime.min
        
        for category, directory in self.subdirs.items():
            category_stats = {
                "files": 0,
                "size_mb": 0,
                "file_types": {},
                "avg_age_hours": 0
            }
            
            try:
                file_ages = []
                
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    
                    if os.path.isfile(filepath):
                        # File count and size
                        category_stats["files"] += 1
                        file_size = os.path.getsize(filepath) / (1024*1024)
                        category_stats["size_mb"] += file_size
                        
                        # File type
                        ext = os.path.splitext(filename)[1]
                        category_stats["file_types"][ext] = category_stats["file_types"].get(ext, 0) + 1
                        
                        # Age tracking
                        mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        age_hours = (datetime.now() - mod_time).total_seconds() / 3600
                        file_ages.append(age_hours)
                        
                        if mod_time < oldest_time:
                            oldest_time = mod_time
                            stats["oldest_file"] = {"category": category, "filename": filename, "age_hours": age_hours}
                        
                        if mod_time > newest_time:
                            newest_time = mod_time
                            stats["newest_file"] = {"category": category, "filename": filename, "age_hours": age_hours}
                
                if file_ages:
                    category_stats["avg_age_hours"] = sum(file_ages) / len(file_ages)
                
            except Exception as e:
                category_stats["error"] = str(e)
            
            stats["categories"][category] = category_stats
            stats["total_files"] += category_stats["files"]
            stats["total_size_mb"] += category_stats["size_mb"]
        
        # Cache health assessment
        if stats["total_size_mb"] > self.cache_settings["max_cache_size_mb"]:
            stats["cache_health"] = "oversized"
        elif stats["total_files"] == 0:
            stats["cache_health"] = "empty"
        elif stats["oldest_file"] and stats["oldest_file"]["age_hours"] > 24*7:  # Older than 1 week
            stats["cache_health"] = "stale"
        
        return stats
    
    def create_session_directory(self, session_id: str) -> str:
        """
        Create a temporary directory for user session data
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Path to session directory
        """
        session_dir = os.path.join(self.subdirs["sessions"], f"session_{session_id}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Create session metadata
        session_metadata = {
            "session_id": session_id,
            "created": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(session_dir, "session_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(session_metadata, f, indent=2)
        
        logger.debug(f"📁 Created session directory: {session_id}")
        return session_dir
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """
        Clean up old session directories
        
        Args:
            max_age_hours: Remove sessions older than this many hours
            
        Returns:
            Dict with cleanup results
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleanup_results = {
            "removed_sessions": [],
            "errors": [],
            "space_freed_mb": 0
        }
        
        try:
            for session_dirname in os.listdir(self.subdirs["sessions"]):
                session_path = os.path.join(self.subdirs["sessions"], session_dirname)
                
                if os.path.isdir(session_path) and session_dirname.startswith("session_"):
                    # Check session age
                    metadata_file = os.path.join(session_path, "session_metadata.json")
                    
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            created_time = datetime.fromisoformat(metadata["created"])
                            
                            if created_time < cutoff_time:
                                # Calculate directory size
                                dir_size = sum(
                                    os.path.getsize(os.path.join(dirpath, filename))
                                    for dirpath, dirnames, filenames in os.walk(session_path)
                                    for filename in filenames
                                ) / (1024*1024)
                                
                                # Remove directory
                                shutil.rmtree(session_path)
                                
                                cleanup_results["removed_sessions"].append({
                                    "session_id": metadata["session_id"],
                                    "age_hours": (datetime.now() - created_time).total_seconds() / 3600,
                                    "size_mb": dir_size
                                })
                                cleanup_results["space_freed_mb"] += dir_size
                                
                        except Exception as e:
                            cleanup_results["errors"].append(f"Error processing {session_dirname}: {e}")
                    else:
                        # No metadata file, remove if old enough based on directory modification time
                        dir_mod_time = datetime.fromtimestamp(os.path.getmtime(session_path))
                        if dir_mod_time < cutoff_time:
                            try:
                                shutil.rmtree(session_path)
                                cleanup_results["removed_sessions"].append({
                                    "session_id": session_dirname,
                                    "age_hours": (datetime.now() - dir_mod_time).total_seconds() / 3600,
                                    "size_mb": 0
                                })
                            except Exception as e:
                                cleanup_results["errors"].append(f"Error removing {session_dirname}: {e}")
        
        except Exception as e:
            cleanup_results["errors"].append(f"Error during session cleanup: {e}")
        
        logger.info(f"🧹 Session cleanup: removed {len(cleanup_results['removed_sessions'])} sessions")
        return cleanup_results
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load data manager metadata"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load data metadata: {e}")
        
        return {"cache_entries": {}}
    
    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save data manager metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save data metadata: {e}")
    
    def _update_cache_metadata(self, category: str, identifier: str, entry_metadata: Dict[str, Any]) -> None:
        """Update metadata for a cache entry"""
        metadata = self.load_metadata()
        
        if "cache_entries" not in metadata:
            metadata["cache_entries"] = {}
        
        entry_key = f"{category}_{identifier}"
        metadata["cache_entries"][entry_key] = {
            "category": category,
            "identifier": identifier,
            "created": datetime.now().isoformat(),
            **entry_metadata
        }
        
        self.save_metadata(metadata)
    
    def export_cache_summary(self, output_file: str) -> Dict[str, Any]:
        """
        Export cache summary to file for analysis
        
        Args:
            output_file: Path to output file
            
        Returns:
            Dict with export results
        """
        try:
            stats = self.get_cache_statistics()
            metadata = self.load_metadata()
            
            summary = {
                "export_timestamp": datetime.now().isoformat(),
                "cache_statistics": stats,
                "metadata": metadata,
                "cache_settings": {
                    setting: str(value) for setting, value in self.cache_settings.items()
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"📋 Cache summary exported to {output_file}")
            return {"success": True, "file": output_file}
            
        except Exception as e:
            logger.error(f"Failed to export cache summary: {e}")
            return {"success": False, "error": str(e)}


def initialize_data_system(data_dir: str = "./ai_models/stock_model/data") -> Dict[str, Any]:
    """
    Initialize the data management system
    
    Args:
        data_dir: Base directory for data storage
        
    Returns:
        Dict with initialization results
    """
    logger.info("🚀 Initializing data management system...")
    
    try:
        # Create data manager
        manager = DataManager(data_dir)
        
        # Get current statistics
        stats = manager.get_cache_statistics()
        
        # Perform initial cleanup if needed
        cleanup_results = {"removed_files": [], "space_freed_mb": 0}
        if stats["cache_health"] in ["oversized", "stale"]:
            cleanup_results = manager.cleanup_old_data(max_age_days=7)
        
        # Clean up old sessions
        session_cleanup = manager.cleanup_old_sessions(max_age_hours=24)
        
        logger.info(f"✅ Data system initialized")
        logger.info(f"📊 Cache status: {stats['total_files']} files, {stats['total_size_mb']:.1f}MB")
        
        return {
            "success": True,
            "manager": manager,
            "initial_stats": stats,
            "cleanup_results": cleanup_results,
            "session_cleanup": session_cleanup
        }
        
    except Exception as e:
        logger.error(f"❌ Data system initialization failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_data_manager(data_dir: str = "./ai_models/stock_model/data") -> DataManager:
    """
    Get a configured data manager instance
    
    Args:
        data_dir: Data directory path
        
    Returns:
        DataManager instance
    """
    return DataManager(data_dir)