"""SQLite connection utilities with context manager support."""
import sqlite3
from pathlib import Path
from typing import Optional


class SQLiteConnection:
    """SQLite database connection with context manager support."""
    
    def __init__(self, db_path: Path):
        """Initialize SQLite connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
    
    def __enter__(self):
        """Enter context manager - establish connection."""
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row  # Enable dict-like row access
        self.cursor = self.connection.cursor()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - close connection."""
        if self.connection:
            if exc_type is None:
                # No exception occurred, commit changes
                self.connection.commit()
            else:
                # Exception occurred, rollback changes
                self.connection.rollback()
            
            self.connection.close()
        
        return False  # Don't suppress exceptions
    
    def execute(self, query: str, params: tuple = ()):
        """Execute a SQL query.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Cursor object
        """
        if not self.cursor:
            raise RuntimeError("Connection not established. Use with statement.")
        
        return self.cursor.execute(query, params)
    
    def executemany(self, query: str, params_list: list):
        """Execute a SQL query with multiple parameter sets.
        
        Args:
            query: SQL query to execute
            params_list: List of parameter tuples
            
        Returns:
            Cursor object
        """
        if not self.cursor:
            raise RuntimeError("Connection not established. Use with statement.")
        
        return self.cursor.executemany(query, params_list)
    
    def fetchall(self):
        """Fetch all rows from the last query."""
        if not self.cursor:
            raise RuntimeError("Connection not established. Use with statement.")
        
        return self.cursor.fetchall()
    
    def fetchone(self):
        """Fetch one row from the last query."""
        if not self.cursor:
            raise RuntimeError("Connection not established. Use with statement.")
        
        return self.cursor.fetchone()
    
    def commit(self):
        """Manually commit changes."""
        if not self.connection:
            raise RuntimeError("Connection not established. Use with statement.")
        
        self.connection.commit()
