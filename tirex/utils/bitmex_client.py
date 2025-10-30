# -*- coding: utf-8 -*-
"""
BitMEX HTTP Client Module

This module provides HTTP client functionality for BitMEX API with retry logic,
error handling, and authentication support using the Strategy pattern.
"""

import time
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps

import requests
from requests.auth import AuthBase

# Setup logging
logger = logging.getLogger(__name__)


# Custom Exception Types
class BitMEXError(Exception):
    """Base exception for BitMEX-related errors."""
    pass


class BitMEXAuthenticationError(BitMEXError):
    """Raised when authentication fails (401 status)."""
    pass


class BitMEXRateLimitError(BitMEXError):
    """Raised when rate limit is exceeded (429 status)."""
    pass


class BitMEXServerError(BitMEXError):
    """Raised when BitMEX server encounters an error (5xx status)."""
    pass


class BitMEXTimeoutError(BitMEXError):
    """Raised when a request times out after retries."""
    pass


class BitMEXConnectionError(BitMEXError):
    """Raised when connection to BitMEX fails after retries."""
    pass


def exponential_backoff_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0
):
    """
    Decorator implementing exponential backoff retry logic.
    
    This decorator retries a function with exponentially increasing delays between
    attempts. It's useful for handling transient network errors.
    
    Parameters
    ----------
    max_retries : int, default=3
        Maximum number of retry attempts
    initial_delay : float, default=1.0
        Initial delay in seconds before first retry
    exponential_base : float, default=2.0
        Base for exponential backoff calculation
    max_delay : float, default=60.0
        Maximum delay between retries in seconds
        
    Returns
    -------
    Callable
        Decorated function with retry logic
        
    Examples
    --------
    >>> @exponential_backoff_retry(max_retries=3, initial_delay=1.0)
    ... def fetch_data():
    ...     return requests.get("https://api.example.com/data")
    
    Notes
    -----
    Delay calculation: delay = min(initial_delay * (exponential_base ** attempt), max_delay)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed.")
            
            # All retries exhausted
            if isinstance(last_exception, requests.exceptions.Timeout):
                raise BitMEXTimeoutError(
                    f"Request timed out after {max_retries} attempts"
                ) from last_exception
            else:
                raise BitMEXConnectionError(
                    f"Connection failed after {max_retries} attempts"
                ) from last_exception
        
        return wrapper
    return decorator


class BitMEXHttpClient:
    """
    HTTP client for BitMEX API with retry logic and error handling.
    
    This client handles low-level HTTP operations including request preparation,
    authentication, error handling, and automatic retries. It uses the Strategy
    pattern for authentication injection.
    
    Parameters
    ----------
    base_url : str
        Base URL for BitMEX API (e.g., 'https://www.bitmex.com/api/v1/')
    authenticator : AuthBase, optional
        Authentication strategy to use for requests
    timeout : int, default=30
        Default timeout for requests in seconds
    max_retries : int, default=3
        Maximum number of retry attempts for failed requests
        
    Attributes
    ----------
    base_url : str
        The base URL for API requests
    authenticator : AuthBase or None
        The authentication strategy being used
    timeout : int
        Default request timeout
    max_retries : int
        Maximum retry attempts
    session : requests.Session
        The HTTP session for connection pooling
        
    Examples
    --------
    >>> from tirex.utils.bitmex_auth import APIKeyAuthWithExpires
    >>> auth = APIKeyAuthWithExpires("my_key", "my_secret")
    >>> client = BitMEXHttpClient(
    ...     base_url="https://www.bitmex.com/api/v1/",
    ...     authenticator=auth
    ... )
    >>> data = client.get("/user")
    
    Notes
    -----
    The client automatically handles:
    - Rate limiting (429 status)
    - Authentication errors (401 status)
    - Server errors (5xx status)
    - Connection timeouts
    - Exponential backoff for retries
    """
    
    # HTTP status code handling configuration
    RATE_LIMIT_DELAY = 25  # seconds
    SERVER_ERROR_DELAY = 1  # seconds
    AUTH_ERROR_MAX_RETRIES = 2
    
    def __init__(
        self,
        base_url: str,
        authenticator: Optional[AuthBase] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize BitMEX HTTP client.
        
        Parameters
        ----------
        base_url : str
            Base URL for BitMEX API
        authenticator : AuthBase, optional
            Authentication strategy (can be injected later)
        timeout : int, default=30
            Default request timeout in seconds
        max_retries : int, default=3
            Maximum retry attempts
            
        Raises
        ------
        AssertionError
            If base_url is empty or timeout/max_retries are invalid
        """
        assert base_url, "Base URL cannot be empty"
        assert timeout > 0, "Timeout must be positive"
        assert max_retries > 0, "Max retries must be positive"
        
        self.base_url = base_url.rstrip('/')
        self.authenticator = authenticator
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = self._initialize_session()
    
    @staticmethod
    def _initialize_session() -> requests.Session:
        """
        Initialize HTTP session with default headers.
        
        Returns
        -------
        requests.Session
            Configured session with default headers
        """
        session = requests.Session()
        session.headers.update({
            'user-agent': 'tirex-bitmex-client/1.0',
            'Content-Type': 'application/json'
        })
        return session
    
    def set_authenticator(self, authenticator: AuthBase) -> None:
        """
        Set or update the authentication strategy.
        
        Parameters
        ----------
        authenticator : AuthBase
            New authentication strategy to use
            
        Examples
        --------
        >>> client = BitMEXHttpClient(base_url="https://api.example.com")
        >>> from tirex.utils.bitmex_auth import APIKeyAuthWithExpires
        >>> auth = APIKeyAuthWithExpires("key", "secret")
        >>> client.set_authenticator(auth)
        """
        self.authenticator = authenticator
    
    def _handle_http_error(self, response: requests.Response, attempt: int) -> None:
        """
        Handle HTTP error responses with appropriate actions.
        
        Parameters
        ----------
        response : requests.Response
            The HTTP response with an error status
        attempt : int
            Current attempt number (0-indexed)
            
        Raises
        ------
        BitMEXAuthenticationError
            For 401 authentication errors
        BitMEXRateLimitError
            For 429 rate limit errors (after delay)
        BitMEXServerError
            For 5xx server errors (after delay)
        BitMEXError
            For other HTTP errors
        """
        status_code = response.status_code
        
        if status_code == 401:
            raise BitMEXAuthenticationError(
                f"Authentication failed: {response.text}"
            )
        
        elif status_code == 429:
            logger.warning(
                f"Rate limit exceeded. Waiting {self.RATE_LIMIT_DELAY}s..."
            )
            time.sleep(self.RATE_LIMIT_DELAY)
            raise BitMEXRateLimitError("Rate limit exceeded")
        
        elif status_code == 503:
            logger.warning(
                f"Service unavailable. Waiting {self.SERVER_ERROR_DELAY}s..."
            )
            time.sleep(self.SERVER_ERROR_DELAY)
            raise BitMEXServerError(f"Service unavailable: {response.text}")
        
        elif 500 <= status_code < 600:
            logger.error(f"Server error {status_code}: {response.text}")
            raise BitMEXServerError(
                f"Server error {status_code}: {response.text}"
            )
        
        else:
            raise BitMEXError(
                f"HTTP {status_code} error: {response.text}"
            )
    
    @exponential_backoff_retry(max_retries=3, initial_delay=1.0)
    def _execute_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute HTTP request with retry logic.
        
        This method handles the low-level request execution with automatic retries
        for transient errors.
        
        Parameters
        ----------
        method : str
            HTTP method (GET, POST, PUT, DELETE)
        path : str
            API endpoint path
        params : dict, optional
            Query parameters
        data : dict, optional
            Request body data
        timeout : int, optional
            Request timeout (uses default if not specified)
            
        Returns
        -------
        dict
            Parsed JSON response
            
        Raises
        ------
        BitMEXAuthenticationError
            For authentication failures
        BitMEXRateLimitError
            For rate limit errors
        BitMEXServerError
            For server errors
        BitMEXTimeoutError
            For timeout after retries
        BitMEXConnectionError
            For connection failures after retries
        """
        url = f"{self.base_url}{path}"
        timeout = timeout or self.timeout
        
        # Prepare request
        req = requests.Request(
            method=method,
            url=url,
            params=params,
            json=data,
            auth=self.authenticator
        )
        
        prepared = self.session.prepare_request(req)
        
        # Execute request
        response = self.session.send(prepared, timeout=timeout)
        
        # Handle HTTP errors
        if not response.ok:
            self._handle_http_error(response, 0)
        
        # Parse and return JSON response
        try:
            return response.json()
        except ValueError:
            return {"status": "success", "data": response.text}
    
    def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute GET request.
        
        Parameters
        ----------
        path : str
            API endpoint path
        params : dict, optional
            Query parameters
        timeout : int, optional
            Request timeout
            
        Returns
        -------
        dict
            Parsed JSON response
        """
        return self._execute_request('GET', path, params=params, timeout=timeout)
    
    def post(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute POST request.
        
        Parameters
        ----------
        path : str
            API endpoint path
        data : dict, optional
            Request body data
        timeout : int, optional
            Request timeout
            
        Returns
        -------
        dict
            Parsed JSON response
        """
        return self._execute_request('POST', path, data=data, timeout=timeout)
    
    def put(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute PUT request.
        
        Parameters
        ----------
        path : str
            API endpoint path
        data : dict, optional
            Request body data
        timeout : int, optional
            Request timeout
            
        Returns
        -------
        dict
            Parsed JSON response
        """
        return self._execute_request('PUT', path, data=data, timeout=timeout)
    
    def delete(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute DELETE request.
        
        Parameters
        ----------
        path : str
            API endpoint path
        params : dict, optional
            Query parameters
        timeout : int, optional
            Request timeout
            
        Returns
        -------
        dict
            Parsed JSON response
        """
        return self._execute_request('DELETE', path, params=params, timeout=timeout)
    
    def close(self) -> None:
        """
        Close the HTTP session and release resources.
        
        Examples
        --------
        >>> client = BitMEXHttpClient(base_url="https://api.example.com")
        >>> # ... use client ...
        >>> client.close()
        """
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes session."""
        self.close()
