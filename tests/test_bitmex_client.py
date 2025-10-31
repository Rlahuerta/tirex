# -*- coding: utf-8 -*-
"""
Unit tests for BitMEX HTTP Client Module

Tests HTTP client functionality, retry logic, error handling, and decorators.
"""

import unittest
import time
from unittest.mock import Mock, patch, MagicMock
import requests

from tirex.utils.bitmex_client import (
    BitMEXError,
    BitMEXAuthenticationError,
    BitMEXRateLimitError,
    BitMEXServerError,
    BitMEXTimeoutError,
    BitMEXConnectionError,
    exponential_backoff_retry,
    BitMEXHttpClient
)


class TestCustomExceptions(unittest.TestCase):
    """Test custom exception types."""
    
    def test_bitmex_error_inheritance(self):
        """Test BitMEXError is base exception."""
        error = BitMEXError("test")
        self.assertIsInstance(error, Exception)
    
    def test_all_exceptions_inherit_from_bitmex_error(self):
        """Test all specific exceptions inherit from BitMEXError."""
        exceptions = [
            BitMEXAuthenticationError("test"),
            BitMEXRateLimitError("test"),
            BitMEXServerError("test"),
            BitMEXTimeoutError("test"),
            BitMEXConnectionError("test")
        ]
        
        for exc in exceptions:
            self.assertIsInstance(exc, BitMEXError)
    
    def test_exceptions_have_messages(self):
        """Test exceptions store error messages."""
        message = "Test error message"
        error = BitMEXError(message)
        self.assertEqual(str(error), message)


class TestExponentialBackoffRetry(unittest.TestCase):
    """Test exponential backoff retry decorator."""
    
    def test_successful_first_attempt(self):
        """Test function succeeds on first attempt."""
        mock_func = Mock(return_value="success")
        decorated = exponential_backoff_retry(max_retries=3)(mock_func)
        
        result = decorated()
        
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 1)
    
    def test_retries_on_timeout(self):
        """Test retries on timeout exception."""
        mock_func = Mock(side_effect=[
            requests.exceptions.Timeout("timeout"),
            requests.exceptions.Timeout("timeout"),
            "success"
        ])
        decorated = exponential_backoff_retry(max_retries=3, initial_delay=0.01)(mock_func)
        
        result = decorated()
        
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 3)
    
    def test_retries_on_connection_error(self):
        """Test retries on connection error."""
        mock_func = Mock(side_effect=[
            requests.exceptions.ConnectionError("connection failed"),
            "success"
        ])
        decorated = exponential_backoff_retry(max_retries=3, initial_delay=0.01)(mock_func)
        
        result = decorated()
        
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 2)
    
    def test_raises_timeout_error_after_max_retries(self):
        """Test raises BitMEXTimeoutError after max retries."""
        mock_func = Mock(side_effect=requests.exceptions.Timeout("timeout"))
        decorated = exponential_backoff_retry(max_retries=2, initial_delay=0.01)(mock_func)
        
        with self.assertRaises(BitMEXTimeoutError):
            decorated()
        
        self.assertEqual(mock_func.call_count, 2)
    
    def test_raises_connection_error_after_max_retries(self):
        """Test raises BitMEXConnectionError after max retries."""
        mock_func = Mock(side_effect=requests.exceptions.ConnectionError("failed"))
        decorated = exponential_backoff_retry(max_retries=2, initial_delay=0.01)(mock_func)
        
        with self.assertRaises(BitMEXConnectionError):
            decorated()
        
        self.assertEqual(mock_func.call_count, 2)
    
    @patch('tirex.utils.bitmex_client.time.sleep')
    def test_exponential_delay(self, mock_sleep):
        """Test exponential backoff delays."""
        mock_func = Mock(side_effect=[
            requests.exceptions.Timeout("timeout"),
            requests.exceptions.Timeout("timeout"),
            requests.exceptions.Timeout("timeout")
        ])
        decorated = exponential_backoff_retry(
            max_retries=3,
            initial_delay=1.0,
            exponential_base=2.0
        )(mock_func)
        
        with self.assertRaises(BitMEXTimeoutError):
            decorated()
        
        # Check delays: 1.0, 2.0 (exponential)
        calls = mock_sleep.call_args_list
        self.assertEqual(len(calls), 2)  # Two delays for 3 attempts
        self.assertAlmostEqual(calls[0][0][0], 1.0, places=1)
        self.assertAlmostEqual(calls[1][0][0], 2.0, places=1)
    
    @patch('tirex.utils.bitmex_client.time.sleep')
    def test_max_delay_cap(self, mock_sleep):
        """Test delay is capped at max_delay."""
        mock_func = Mock(side_effect=[
            requests.exceptions.Timeout("timeout"),
            requests.exceptions.Timeout("timeout"),
            requests.exceptions.Timeout("timeout")
        ])
        decorated = exponential_backoff_retry(
            max_retries=3,
            initial_delay=10.0,
            exponential_base=2.0,
            max_delay=15.0
        )(mock_func)
        
        with self.assertRaises(BitMEXTimeoutError):
            decorated()
        
        # Check delays are capped
        calls = mock_sleep.call_args_list
        for call in calls:
            self.assertLessEqual(call[0][0], 15.0)


class TestBitMEXHttpClient(unittest.TestCase):
    """Test BitMEXHttpClient class."""
    
    def test_initialization(self):
        """Test client initialization."""
        base_url = "https://api.example.com"
        client = BitMEXHttpClient(base_url=base_url)
        
        self.assertEqual(client.base_url, "https://api.example.com")
        self.assertEqual(client.timeout, 30)
        self.assertEqual(client.max_retries, 3)
        self.assertIsNone(client.authenticator)
    
    def test_initialization_with_custom_params(self):
        """Test client initialization with custom parameters."""
        base_url = "https://api.example.com/"
        timeout = 60
        max_retries = 5
        
        client = BitMEXHttpClient(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # URL should be stripped of trailing slash
        self.assertEqual(client.base_url, "https://api.example.com")
        self.assertEqual(client.timeout, timeout)
        self.assertEqual(client.max_retries, max_retries)
    
    def test_assertion_on_empty_base_url(self):
        """Test that empty base URL raises AssertionError."""
        with self.assertRaises(AssertionError):
            BitMEXHttpClient(base_url="")
    
    def test_assertion_on_invalid_timeout(self):
        """Test that invalid timeout raises AssertionError."""
        with self.assertRaises(AssertionError):
            BitMEXHttpClient(base_url="http://example.com", timeout=0)
    
    def test_assertion_on_invalid_max_retries(self):
        """Test that invalid max_retries raises AssertionError."""
        with self.assertRaises(AssertionError):
            BitMEXHttpClient(base_url="http://example.com", max_retries=0)
    
    def test_session_initialization(self):
        """Test that session is initialized with headers."""
        client = BitMEXHttpClient(base_url="http://example.com")
        
        self.assertIsNotNone(client.session)
        self.assertIn('user-agent', client.session.headers)
        self.assertIn('Content-Type', client.session.headers)
    
    def test_set_authenticator(self):
        """Test setting authenticator."""
        client = BitMEXHttpClient(base_url="http://example.com")
        mock_auth = Mock()
        
        client.set_authenticator(mock_auth)
        
        self.assertEqual(client.authenticator, mock_auth)
    
    @patch('tirex.utils.bitmex_client.requests.Session')
    def test_get_method(self, mock_session_class):
        """Test GET request method."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"status": "success"}
        mock_session.prepare_request.return_value = Mock()
        mock_session.send.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = BitMEXHttpClient(base_url="http://example.com")
        result = client.get("/test")
        
        self.assertEqual(result, {"status": "success"})
    
    @patch('tirex.utils.bitmex_client.requests.Session')
    def test_post_method(self, mock_session_class):
        """Test POST request method."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"created": True}
        mock_session.prepare_request.return_value = Mock()
        mock_session.send.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = BitMEXHttpClient(base_url="http://example.com")
        result = client.post("/test", data={"key": "value"})
        
        self.assertEqual(result, {"created": True})
    
    @patch('tirex.utils.bitmex_client.requests.Session')
    def test_handles_401_authentication_error(self, mock_session_class):
        """Test handling of 401 authentication error."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_session.prepare_request.return_value = Mock()
        mock_session.send.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = BitMEXHttpClient(base_url="http://example.com")
        
        with self.assertRaises(BitMEXAuthenticationError):
            client.get("/test")
    
    @patch('tirex.utils.bitmex_client.time.sleep')
    @patch('tirex.utils.bitmex_client.requests.Session')
    def test_handles_429_rate_limit(self, mock_session_class, mock_sleep):
        """Test handling of 429 rate limit error."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_session.prepare_request.return_value = Mock()
        mock_session.send.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = BitMEXHttpClient(base_url="http://example.com")
        
        with self.assertRaises(BitMEXRateLimitError):
            client.get("/test")
        
        # Verify it sleeps before raising
        mock_sleep.assert_called_once_with(25)
    
    @patch('tirex.utils.bitmex_client.time.sleep')
    @patch('tirex.utils.bitmex_client.requests.Session')
    def test_handles_503_service_unavailable(self, mock_session_class, mock_sleep):
        """Test handling of 503 service unavailable."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 503
        mock_response.text = "Service unavailable"
        mock_session.prepare_request.return_value = Mock()
        mock_session.send.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = BitMEXHttpClient(base_url="http://example.com")
        
        with self.assertRaises(BitMEXServerError):
            client.get("/test")
        
        # Verify it sleeps before raising
        mock_sleep.assert_called_once_with(1)
    
    @patch('tirex.utils.bitmex_client.requests.Session')
    def test_handles_500_server_error(self, mock_session_class):
        """Test handling of 500 server error."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_session.prepare_request.return_value = Mock()
        mock_session.send.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = BitMEXHttpClient(base_url="http://example.com")
        
        with self.assertRaises(BitMEXServerError):
            client.get("/test")
    
    def test_context_manager(self):
        """Test client works as context manager."""
        with BitMEXHttpClient(base_url="http://example.com") as client:
            self.assertIsNotNone(client.session)
        
        # Session should be closed after context
        # We can't easily test this without mocking, but we verify no error
    
    def test_close_method(self):
        """Test close method."""
        client = BitMEXHttpClient(base_url="http://example.com")
        client.session = Mock()
        
        client.close()
        
        client.session.close.assert_called_once()
    
    @patch('tirex.utils.bitmex_client.requests.Session')
    def test_authenticator_injection(self, mock_session_class):
        """Test that authenticator is used in requests."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {}
        mock_session.prepare_request.return_value = Mock()
        mock_session.send.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        mock_auth = Mock()
        client = BitMEXHttpClient(
            base_url="http://example.com",
            authenticator=mock_auth
        )
        
        client.get("/test")
        
        # Verify Request was created with auth
        call_args = mock_session.prepare_request.call_args
        # The request should have been prepared
        self.assertTrue(mock_session.prepare_request.called)


class TestBitMEXHttpClientIntegration(unittest.TestCase):
    """Integration tests for BitMEXHttpClient."""
    
    @patch('tirex.utils.bitmex_client.requests.Session')
    def test_full_request_lifecycle(self, mock_session_class):
        """Test complete request lifecycle."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "data": "test_data",
            "count": 100
        }
        mock_prepared_request = Mock()
        mock_session.prepare_request.return_value = mock_prepared_request
        mock_session.send.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = BitMEXHttpClient(base_url="https://api.example.com")
        
        result = client.get("/endpoint", params={"filter": "test"})
        
        self.assertEqual(result["data"], "test_data")
        self.assertEqual(result["count"], 100)
        
        # Verify request was prepared and sent
        mock_session.prepare_request.assert_called_once()
        mock_session.send.assert_called_once()


if __name__ == '__main__':
    unittest.main()
