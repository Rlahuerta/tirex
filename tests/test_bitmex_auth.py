# -*- coding: utf-8 -*-
"""
Unit tests for BitMEX Authentication Module

Tests authentication strategies, signature generation, and Protocol compliance.
"""

import unittest
import time
from unittest.mock import Mock, patch
# import hmac
# import hashlib

from tirex.utils.bitmex_auth import (
    generate_expires,
    generate_signature,
    AccessTokenAuth,
    APIKeyAuth,
    APIKeyAuthWithExpires,
    # BitMEXAuthenticator
)


class TestGenerateExpires(unittest.TestCase):
    """Test generate_expires function."""
    
    def test_returns_future_timestamp(self):
        """Test that generated timestamp is in the future."""
        expires = generate_expires(3600)
        current = int(time.time())
        self.assertGreater(expires, current)
        self.assertLessEqual(expires, current + 3601)
    
    def test_custom_offset(self):
        """Test custom offset parameter."""
        offset = 300  # 5 minutes
        expires = generate_expires(offset)
        current = int(time.time())
        expected_min = current + offset - 1
        expected_max = current + offset + 1
        self.assertGreaterEqual(expires, expected_min)
        self.assertLessEqual(expires, expected_max)
    
    def test_assertion_on_negative_offset(self):
        """Test that negative offset raises AssertionError."""
        with self.assertRaises(AssertionError):
            generate_expires(-100)
    
    def test_assertion_on_zero_offset(self):
        """Test that zero offset raises AssertionError."""
        with self.assertRaises(AssertionError):
            generate_expires(0)


class TestGenerateSignature(unittest.TestCase):
    """Test generate_signature function."""
    
    def test_signature_format(self):
        """Test that signature is 64-character hexadecimal string."""
        secret = "test_secret"
        verb = "GET"
        url = "https://www.bitmex.com/api/v1/user"
        nonce = 1609459200
        data = ""
        
        signature = generate_signature(secret, verb, url, nonce, data)
        
        self.assertEqual(len(signature), 64)
        self.assertTrue(all(c in '0123456789abcdef' for c in signature))
    
    def test_signature_deterministic(self):
        """Test that same inputs always produce same signature."""
        secret = "my_secret"
        verb = "POST"
        url = "https://www.bitmex.com/api/v1/order"
        nonce = 1234567890
        data = '{"symbol":"XBTUSD"}'
        
        sig1 = generate_signature(secret, verb, url, nonce, data)
        sig2 = generate_signature(secret, verb, url, nonce, data)
        
        self.assertEqual(sig1, sig2)
    
    def test_different_inputs_different_signatures(self):
        """Test that different inputs produce different signatures."""
        secret = "secret"
        verb1 = "GET"
        verb2 = "POST"
        url = "https://www.bitmex.com/api/v1/user"
        nonce = 1234567890
        data = ""
        
        sig1 = generate_signature(secret, verb1, url, nonce, data)
        sig2 = generate_signature(secret, verb2, url, nonce, data)
        
        self.assertNotEqual(sig1, sig2)
    
    def test_handles_query_parameters(self):
        """Test signature generation with query parameters."""
        secret = "secret"
        verb = "GET"
        url = "https://www.bitmex.com/api/v1/trade?symbol=XBTUSD&count=100"
        nonce = 1234567890
        data = ""
        
        signature = generate_signature(secret, verb, url, nonce, data)
        
        self.assertEqual(len(signature), 64)
    
    def test_handles_bytes_data(self):
        """Test signature generation with bytes data."""
        secret = "secret"
        verb = "POST"
        url = "https://www.bitmex.com/api/v1/order"
        nonce = 1234567890
        data = b'{"symbol":"XBTUSD"}'
        
        signature = generate_signature(secret, verb, url, nonce, data)
        
        self.assertEqual(len(signature), 64)
    
    def test_assertion_on_empty_secret(self):
        """Test that empty secret raises AssertionError."""
        with self.assertRaises(AssertionError):
            generate_signature("", "GET", "http://example.com", 123, "")
    
    def test_assertion_on_invalid_verb(self):
        """Test that invalid HTTP verb raises AssertionError."""
        with self.assertRaises(AssertionError):
            generate_signature("secret", "INVALID", "http://example.com", 123, "")
    
    def test_all_valid_verbs(self):
        """Test that all valid HTTP verbs work."""
        secret = "secret"
        url = "https://www.bitmex.com/api/v1/user"
        nonce = 1234567890
        data = ""
        
        valid_verbs = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
        
        for verb in valid_verbs:
            signature = generate_signature(secret, verb, url, nonce, data)
            self.assertEqual(len(signature), 64)


class TestAccessTokenAuth(unittest.TestCase):
    """Test AccessTokenAuth class."""
    
    def test_initialization(self):
        """Test AccessTokenAuth initialization."""
        token = "test_token_12345"
        auth = AccessTokenAuth(token)
        
        self.assertEqual(auth.token, token)
    
    def test_adds_header_to_request(self):
        """Test that auth adds access-token header."""
        token = "test_token"
        auth = AccessTokenAuth(token)
        
        # Create mock request
        request = Mock()
        request.headers = {}
        
        # Apply auth
        result = auth(request)
        
        self.assertIn('access-token', result.headers)
        self.assertEqual(result.headers['access-token'], token)
    
    def test_no_header_if_no_token(self):
        """Test that no header added if token is None."""
        auth = AccessTokenAuth(None)
        
        request = Mock()
        request.headers = {}
        
        result = auth(request)
        
        self.assertNotIn('access-token', result.headers)
    
    def test_returns_request(self):
        """Test that auth returns the request object."""
        auth = AccessTokenAuth("token")
        request = Mock()
        request.headers = {}
        
        result = auth(request)
        
        self.assertIs(result, request)


class TestAPIKeyAuth(unittest.TestCase):
    """Test APIKeyAuth class."""
    
    def test_initialization(self):
        """Test APIKeyAuth initialization."""
        api_key = "test_key"
        api_secret = "test_secret"
        
        auth = APIKeyAuth(api_key, api_secret)
        
        self.assertEqual(auth.apiKey, api_key)
        self.assertEqual(auth.apiSecret, api_secret)
    
    def test_assertion_on_empty_key(self):
        """Test that empty API key raises AssertionError."""
        with self.assertRaises(AssertionError):
            APIKeyAuth("", "secret")
    
    def test_assertion_on_empty_secret(self):
        """Test that empty API secret raises AssertionError."""
        with self.assertRaises(AssertionError):
            APIKeyAuth("key", "")
    
    @patch('tirex.utils.bitmex_auth.generate_expires')
    @patch('tirex.utils.bitmex_auth.generate_signature')
    def test_adds_auth_headers(self, mock_signature, mock_expires):
        """Test that auth adds required headers."""
        mock_expires.return_value = 1234567890
        mock_signature.return_value = "a" * 64
        
        auth = APIKeyAuth("test_key", "test_secret")
        
        request = Mock()
        request.headers = {}
        request.method = "GET"
        request.url = "https://www.bitmex.com/api/v1/user"
        request.body = None
        
        result = auth(request)
        
        self.assertIn('api-expires', result.headers)
        self.assertIn('api-key', result.headers)
        self.assertIn('api-signature', result.headers)
        self.assertEqual(result.headers['api-key'], "test_key")
    
    @patch('tirex.utils.bitmex_auth.generate_expires')
    def test_uses_generate_expires(self, mock_expires):
        """Test that APIKeyAuth uses generate_expires."""
        mock_expires.return_value = 9999999999
        
        auth = APIKeyAuth("key", "secret")
        request = Mock()
        request.headers = {}
        request.method = "GET"
        request.url = "http://example.com"
        request.body = None
        
        auth(request)
        
        mock_expires.assert_called_once()
        self.assertEqual(request.headers['api-expires'], "9999999999")


class TestAPIKeyAuthWithExpires(unittest.TestCase):
    """Test APIKeyAuthWithExpires class."""
    
    def test_initialization(self):
        """Test APIKeyAuthWithExpires initialization."""
        api_key = "test_key"
        api_secret = "test_secret"
        
        auth = APIKeyAuthWithExpires(api_key, api_secret)
        
        self.assertEqual(auth.apiKey, api_key)
        self.assertEqual(auth.apiSecret, api_secret)
    
    def test_assertion_on_empty_key(self):
        """Test that empty API key raises AssertionError."""
        with self.assertRaises(AssertionError):
            APIKeyAuthWithExpires("", "secret")
    
    def test_assertion_on_empty_secret(self):
        """Test that empty API secret raises AssertionError."""
        with self.assertRaises(AssertionError):
            APIKeyAuthWithExpires("key", "")
    
    @patch('tirex.utils.bitmex_auth.time.time')
    @patch('tirex.utils.bitmex_auth.generate_signature')
    def test_adds_auth_headers(self, mock_signature, mock_time):
        """Test that auth adds required headers with expires."""
        mock_time.return_value = 1000000000.0
        mock_signature.return_value = "b" * 64
        
        auth = APIKeyAuthWithExpires("test_key", "test_secret")
        
        request = Mock()
        request.headers = {}
        request.method = "GET"
        request.url = "https://www.bitmex.com/api/v1/user"
        request.body = None
        
        result = auth(request)
        
        self.assertIn('api-expires', result.headers)
        self.assertIn('api-key', result.headers)
        self.assertIn('api-signature', result.headers)
        
        # Verify expires is current time + 5 seconds
        expires = int(result.headers['api-expires'])
        expected = int(round(1000000000.0) + 5)
        self.assertEqual(expires, expected)
    
    @patch('tirex.utils.bitmex_auth.time.time')
    def test_uses_5_second_grace_period(self, mock_time):
        """Test that auth uses 5-second grace period."""
        current_time = 2000000000.0
        mock_time.return_value = current_time
        
        auth = APIKeyAuthWithExpires("key", "secret")
        request = Mock()
        request.headers = {}
        request.method = "GET"
        request.url = "http://example.com"
        request.body = None
        
        auth(request)
        
        expires = int(request.headers['api-expires'])
        expected = int(round(current_time) + 5)
        self.assertEqual(expires, expected)
    
    def test_returns_request(self):
        """Test that auth returns the request object."""
        auth = APIKeyAuthWithExpires("key", "secret")
        request = Mock()
        request.headers = {}
        request.method = "GET"
        request.url = "http://example.com"
        request.body = None
        
        result = auth(request)
        
        self.assertIs(result, request)


class TestBitMEXAuthenticatorProtocol(unittest.TestCase):
    """Test that authentication classes comply with Protocol."""
    
    def test_access_token_auth_is_callable(self):
        """Test AccessTokenAuth is callable."""
        auth = AccessTokenAuth("token")
        self.assertTrue(callable(auth))
    
    def test_api_key_auth_is_callable(self):
        """Test APIKeyAuth is callable."""
        auth = APIKeyAuth("key", "secret")
        self.assertTrue(callable(auth))
    
    def test_api_key_auth_with_expires_is_callable(self):
        """Test APIKeyAuthWithExpires is callable."""
        auth = APIKeyAuthWithExpires("key", "secret")
        self.assertTrue(callable(auth))
    
    def test_all_auths_have_call_method(self):
        """Test all authenticators have __call__ method."""
        auths = [
            AccessTokenAuth("token"),
            APIKeyAuth("key", "secret"),
            APIKeyAuthWithExpires("key", "secret")
        ]
        
        for auth in auths:
            self.assertTrue(hasattr(auth, '__call__'))


class TestSignatureIntegration(unittest.TestCase):
    """Integration tests for signature generation."""
    
    def test_known_signature(self):
        """Test against a known signature (if available from BitMEX docs)."""
        # This would be a real example from BitMEX API documentation
        # For now, we test the structure
        secret = "LAqUlngMIQkIUjXMUreyu3qn"
        verb = "GET"
        url = "https://www.bitmex.com/api/v1/instrument"
        nonce = 1518064236
        data = ""
        
        signature = generate_signature(secret, verb, url, nonce, data)
        
        # Verify it's a valid hex string
        self.assertEqual(len(signature), 64)
        try:
            int(signature, 16)
        except ValueError:
            self.fail("Signature is not valid hexadecimal")


if __name__ == '__main__':
    unittest.main()
