import threading
import time
from pathlib import Path
from typing import Optional, List
from queue import Queue, Empty
from contextlib import contextmanager
import logging

import chromadb
from chromadb import Settings

logger = logging.getLogger(__name__)


class ChromaConnection:
    """Wrapper for a ChromaDB client connection with health tracking."""
    
    def __init__(self, client: chromadb.PersistentClient, created_at: float):
        self.client = client
        self.created_at = created_at
        self.last_used = created_at
        self.is_healthy = True
        self.use_count = 0
    
    def mark_used(self):
        """Mark this connection as recently used."""
        self.last_used = time.time()
        self.use_count += 1
    
    def check_health(self) -> bool:
        """Check if the connection is healthy."""
        try:
            # Simple health check - try to list collections
            self.client.list_collections()
            self.is_healthy = True
            return True
        except Exception as e:
            logger.warning(f"Connection health check failed: {e}")
            self.is_healthy = False
            return False
    
    def close(self):
        """Close the connection."""
        try:
            # ChromaDB PersistentClient doesn't have an explicit close method
            # but we can mark it as unhealthy
            self.is_healthy = False
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")


class ChromaConnectionPool:
    """
    Connection pool for ChromaDB clients.
    
    Manages a pool of ChromaDB client connections with configurable size,
    timeout, and health monitoring capabilities.
    """
    
    def __init__(
        self,
        db_dir: Path,
        pool_size: int = 5,
        max_pool_size: int = 10,
        connection_timeout: float = 30.0,
        health_check_interval: float = 300.0,  # 5 minutes
        max_connection_age: float = 3600.0,    # 1 hour
    ):
        """
        Initialize the connection pool.
        
        Args:
            db_dir: Path to the ChromaDB database directory
            pool_size: Initial number of connections to create
            max_pool_size: Maximum number of connections in the pool
            connection_timeout: Timeout in seconds for getting a connection
            health_check_interval: Interval in seconds between health checks
            max_connection_age: Maximum age in seconds before connection is recycled
        """
        self.db_dir = db_dir
        self.pool_size = pool_size
        self.max_pool_size = max_pool_size
        self.connection_timeout = connection_timeout
        self.health_check_interval = health_check_interval
        self.max_connection_age = max_connection_age
        
        self._pool: Queue[ChromaConnection] = Queue(maxsize=max_pool_size)
        self._lock = threading.RLock()
        self._created_connections = 0
        self._last_health_check = 0.0
        self._closed = False
        
        # Initialize the pool with initial connections
        self._initialize_pool()
    
    def _create_connection(self) -> ChromaConnection:
        """Create a new ChromaDB connection."""
        try:
            client = chromadb.PersistentClient(
                path=self.db_dir,
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                ),
            )
            connection = ChromaConnection(client, time.time())
            logger.debug(f"Created new ChromaDB connection (total: {self._created_connections + 1})")
            return connection
        except Exception as e:
            logger.error(f"Failed to create ChromaDB connection: {e}")
            raise
    
    def _initialize_pool(self):
        """Initialize the pool with the configured number of connections."""
        with self._lock:
            for _ in range(self.pool_size):
                if self._created_connections < self.max_pool_size:
                    try:
                        connection = self._create_connection()
                        self._pool.put_nowait(connection)
                        self._created_connections += 1
                    except Exception as e:
                        logger.error(f"Failed to initialize connection: {e}")
                        break
    
    def _should_recycle_connection(self, connection: ChromaConnection) -> bool:
        """Check if a connection should be recycled due to age or health."""
        current_time = time.time()
        
        # Check age
        if current_time - connection.created_at > self.max_connection_age:
            logger.debug("Connection exceeded max age, recycling")
            return True
        
        # Check health if it's time
        if current_time - self._last_health_check > self.health_check_interval:
            if not connection.check_health():
                logger.debug("Connection failed health check, recycling")
                return True
        
        return False
    
    def _perform_health_checks(self):
        """Perform health checks on connections if needed."""
        current_time = time.time()
        if current_time - self._last_health_check > self.health_check_interval:
            self._last_health_check = current_time
            logger.debug("Performing connection pool health checks")
    
    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool as a context manager.
        
        Yields:
            ChromaConnection: A healthy connection from the pool
            
        Raises:
            RuntimeError: If pool is closed or no connection available within timeout
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        connection = None
        start_time = time.time()
        
        try:
            # Try to get a connection from the pool
            while time.time() - start_time < self.connection_timeout:
                try:
                    connection = self._pool.get(timeout=1.0)
                    
                    # Check if connection should be recycled
                    if self._should_recycle_connection(connection):
                        connection.close()
                        with self._lock:
                            self._created_connections -= 1
                        connection = None
                        continue
                    
                    # Mark connection as used and yield it
                    connection.mark_used()
                    self._perform_health_checks()
                    yield connection
                    return
                    
                except Empty:
                    # No connection available, try to create a new one if under limit
                    with self._lock:
                        if self._created_connections < self.max_pool_size:
                            try:
                                connection = self._create_connection()
                                self._created_connections += 1
                                connection.mark_used()
                                self._perform_health_checks()
                                yield connection
                                return
                            except Exception as e:
                                logger.error(f"Failed to create new connection: {e}")
                                continue
            
            raise RuntimeError(f"Failed to get connection within {self.connection_timeout} seconds")
            
        finally:
            # Return connection to pool if we got one
            if connection and not self._closed:
                try:
                    self._pool.put_nowait(connection)
                except Exception as e:
                    logger.warning(f"Failed to return connection to pool: {e}")
                    # If we can't return it, close it and decrement counter
                    connection.close()
                    with self._lock:
                        self._created_connections -= 1
    
    def get_stats(self) -> dict:
        """Get connection pool statistics."""
        with self._lock:
            return {
                "pool_size": self.pool_size,
                "max_pool_size": self.max_pool_size,
                "current_connections": self._created_connections,
                "available_connections": self._pool.qsize(),
                "connection_timeout": self.connection_timeout,
                "health_check_interval": self.health_check_interval,
                "max_connection_age": self.max_connection_age,
                "last_health_check": self._last_health_check,
                "closed": self._closed,
            }
    
    def close(self):
        """Close all connections in the pool."""
        if self._closed:
            return
        
        self._closed = True
        connections_closed = 0
        
        # Close all connections in the pool
        while not self._pool.empty():
            try:
                connection = self._pool.get_nowait()
                connection.close()
                connections_closed += 1
            except Empty:
                break
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
        
        with self._lock:
            self._created_connections = 0
        
        logger.info(f"Closed {connections_closed} connections in ChromaDB connection pool")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()