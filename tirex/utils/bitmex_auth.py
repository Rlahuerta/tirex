# -*- coding: utf-8 -*-
"""
BitMEX Authentication Module

This module provides authentication strategies for BitMEX API requests.
It implements the Strategy pattern through a Protocol-based interface to support
Dependency Inversion Principle.
"""

import hmac
import hashlib
import time
import urllib.parse
from typing import Protocol
from requests.auth import AuthBase


class BitMEXAuthenticator(Protocol):
    """
    Protocol defining the interface for BitMEX authentication strategies.
    
    This protocol enables Dependency Inversion by allowing different authentication
    implementations to be injected without tight coupling.
    """
    
    def __call__(self, r) -> any:
        """
        Authenticate a request by adding appropriate headers.
        
        Parameters
        ----------
        r : requests.PreparedRequest
            The request object to authenticate
            
        Returns
        -------
        requests.PreparedRequest
            The authenticated request with headers added
        """
        ...


def generate_expires(offset_seconds: int = 3600) -> int:
    """
    Generate an expiration timestamp for API requests.
    
    Parameters
    ----------
    offset_seconds : int, default=3600
        Number of seconds from now until expiration
        
    Returns
    -------
    int
        Unix timestamp for expiration time
        
    Examples
    --------
    >>> expires = generate_expires(3600)
    >>> expires > time.time()
    True
    """
    assert offset_seconds > 0, "Offset must be positive"
    return int(time.time() + offset_seconds)


def generate_signature(secret: str, verb: str, url: str, nonce: int, data: str) -> str:
    """
    Generate HMAC-SHA256 signature for BitMEX API authentication.
    
    This function creates a request signature compatible with BitMEX API requirements.
    The signature is computed from the HTTP verb, URL path, nonce, and request body.
    
    Parameters
    ----------
    secret : str
        API secret key for HMAC computation
    verb : str
        HTTP method (GET, POST, PUT, DELETE)
    url : str
        Full request URL including query parameters
    nonce : int
        Unix timestamp or unique nonce for request
    data : str or bytes
        Request body data (empty string for GET requests)
        
    Returns
    -------
    str
        Hexadecimal HMAC-SHA256 signature
        
    Raises
    ------
    AssertionError
        If secret is empty or verb is not a valid HTTP method
        
    Examples
    --------
    >>> secret = "my_secret_key"
    >>> verb = "GET"
    >>> url = "https://www.bitmex.com/api/v1/user"
    >>> nonce = 1609459200
    >>> signature = generate_signature(secret, verb, url, nonce, "")
    >>> len(signature) == 64  # SHA256 hex digest length
    True
    
    Notes
    -----
    The signature is computed as HMAC-SHA256(secret, message) where:
    message = verb + path + nonce + data
    
    For more details, see: https://www.bitmex.com/app/apiKeysUsage
    """
    assert secret, "API secret cannot be empty"
    assert verb in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'], f"Invalid HTTP verb: {verb}"
    
    # Parse URL to extract path and query
    parsed_url = urllib.parse.urlparse(url)
    path = parsed_url.path
    
    if parsed_url.query:
        path = path + '?' + parsed_url.query
    
    # Handle bytes/bytearray data
    if isinstance(data, (bytes, bytearray)):
        data = data.decode('utf8')
    
    # Construct message for HMAC
    message = verb + path + str(nonce) + data
    
    # Compute HMAC-SHA256 signature
    signature = hmac.new(
        bytes(secret, 'utf8'),
        bytes(message, 'utf8'),
        digestmod=hashlib.sha256
    ).hexdigest()
    
    return signature


class AccessTokenAuth(AuthBase):
    """
    Access token-based authentication for BitMEX API requests.
    
    This authenticator adds an access token header to requests. It's typically used
    for temporary session-based authentication.
    
    Parameters
    ----------
    access_token : str
        The access token obtained from login/authentication
        
    Attributes
    ----------
    token : str
        The stored access token
        
    Examples
    --------
    >>> import requests
    >>> auth = AccessTokenAuth("my_access_token")
    >>> response = requests.get("https://api.example.com/data", auth=auth)
    
    Notes
    -----
    This authentication method is less secure than API key authentication for
    long-term access. Use APIKeyAuthWithExpires for production applications.
    """
    
    def __init__(self, access_token: str):
        """
        Initialize access token authenticator.
        
        Parameters
        ----------
        access_token : str
            The access token for authentication
        """
        self.token = access_token
    
    def __call__(self, r):
        """
        Add access token header to the request.
        
        Parameters
        ----------
        r : requests.PreparedRequest
            The request to authenticate
            
        Returns
        -------
        requests.PreparedRequest
            The request with access-token header added
        """
        if self.token:
            r.headers['access-token'] = self.token
        return r


class APIKeyAuth(AuthBase):
    """
    API Key-based authentication for BitMEX API requests using nonce.
    
    This authenticator uses API key and secret to generate HMAC signatures with
    a nonce-based expiration. For better reliability with concurrent requests,
    consider using APIKeyAuthWithExpires instead.
    
    Parameters
    ----------
    api_key : str
        The API key from BitMEX account
    api_secret : str
        The API secret from BitMEX account
        
    Attributes
    ----------
    apiKey : str
        The stored API key
    apiSecret : str
        The stored API secret
        
    Examples
    --------
    >>> import requests
    >>> auth = APIKeyAuth("my_api_key", "my_api_secret")
    >>> response = requests.get("https://www.bitmex.com/api/v1/user", auth=auth)
    
    See Also
    --------
    APIKeyAuthWithExpires : Recommended for production use
    generate_signature : Signature generation function
    
    Notes
    -----
    This implementation uses a nonce that may cause issues with concurrent requests
    to the same API endpoint. Use APIKeyAuthWithExpires for better reliability.
    """
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize API key authenticator.
        
        Parameters
        ----------
        api_key : str
            The API key from BitMEX
        api_secret : str
            The API secret from BitMEX
            
        Raises
        ------
        AssertionError
            If api_key or api_secret is empty
        """
        assert api_key, "API key cannot be empty"
        assert api_secret, "API secret cannot be empty"
        
        self.apiKey = api_key
        self.apiSecret = api_secret
    
    def __call__(self, r):
        """
        Add API authentication headers to the request.
        
        Generates api-expires, api-key, and api-signature headers using HMAC-SHA256.
        
        Parameters
        ----------
        r : requests.PreparedRequest
            The request to authenticate
            
        Returns
        -------
        requests.PreparedRequest
            The request with authentication headers added
        """
        nonce = generate_expires()
        r.headers['api-expires'] = str(nonce)
        r.headers['api-key'] = self.apiKey
        r.headers['api-signature'] = generate_signature(
            self.apiSecret, r.method, r.url, nonce, r.body or ''
        )
        return r


class APIKeyAuthWithExpires(AuthBase):
    """
    API Key authentication with expires-based timing for BitMEX API.
    
    This is the recommended authentication method for production use. It uses
    an expires timestamp instead of a nonce, preventing collisions when multiple
    processes use the same API key with out-of-order request arrival.
    
    Parameters
    ----------
    api_key : str
        The API key from BitMEX account
    api_secret : str
        The API secret from BitMEX account
        
    Attributes
    ----------
    apiKey : str
        The stored API key
    apiSecret : str
        The stored API secret
        
    Examples
    --------
    >>> import requests
    >>> auth = APIKeyAuthWithExpires("my_api_key", "my_api_secret")
    >>> response = requests.get("https://www.bitmex.com/api/v1/user", auth=auth)
    
    See Also
    --------
    APIKeyAuth : Basic nonce-based authentication
    generate_signature : Signature generation function
    
    Notes
    -----
    This implementation uses a 5-second grace period for clock skew tolerance.
    The expires timestamp prevents request collision issues that can occur with
    nonce-based authentication in distributed systems.
    
    For more details, see: https://www.bitmex.com/app/apiKeys
    """
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize API key authenticator with expires support.
        
        Parameters
        ----------
        api_key : str
            The API key from BitMEX
        api_secret : str
            The API secret from BitMEX
            
        Raises
        ------
        AssertionError
            If api_key or api_secret is empty
        """
        assert api_key, "API key cannot be empty"
        assert api_secret, "API secret cannot be empty"
        
        self.apiKey = api_key
        self.apiSecret = api_secret
    
    def __call__(self, r):
        """
        Add API authentication headers with expires timestamp.
        
        Generates api-expires, api-key, and api-signature headers with a 5-second
        grace period for clock skew.
        
        Parameters
        ----------
        r : requests.PreparedRequest
            The request to authenticate
            
        Returns
        -------
        requests.PreparedRequest
            The request with authentication headers added
        """
        # 5-second grace period for clock skew
        expires = int(round(time.time()) + 5)
        r.headers['api-expires'] = str(expires)
        r.headers['api-key'] = self.apiKey
        r.headers['api-signature'] = generate_signature(
            self.apiSecret, r.method, r.url, expires, r.body or ''
        )
        return r
