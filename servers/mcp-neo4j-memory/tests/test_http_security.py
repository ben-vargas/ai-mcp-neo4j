import pytest
import os
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

from mcp_neo4j_memory.server import validate_cypher_identifier


class TestCypherValidation:
    """Test Cypher injection prevention"""
    
    def test_valid_identifiers(self):
        """Test that valid identifiers pass validation"""
        valid_ids = [
            "Person", "Employee", "knows", "works_at", "PART_OF", 
            "User123", "_private", "node_1", "rel-type"
        ]
        for identifier in valid_ids:
            result = validate_cypher_identifier(identifier, "test")
            assert result == identifier
    
    def test_invalid_identifiers(self):
        """Test that invalid identifiers are rejected"""
        invalid_ids = [
            "",  # empty
            "123invalid",  # starts with number
            "DROP TABLE",  # contains space
            "MATCH",  # reserved keyword
            "CREATE",  # reserved keyword
            "node.property",  # contains dot
            "node;DROP",  # contains semicolon
            "node'injection",  # contains quote
            "a" * 65,  # too long
        ]
        for identifier in invalid_ids:
            with pytest.raises(ValueError):
                validate_cypher_identifier(identifier, "test")
    
    def test_reserved_keywords(self):
        """Test that reserved keywords are rejected"""
        keywords = ["MATCH", "CREATE", "DELETE", "MERGE", "SET", "RETURN"]
        for keyword in keywords:
            with pytest.raises(ValueError, match="Reserved keyword"):
                validate_cypher_identifier(keyword, "test")
            # Test case insensitive
            with pytest.raises(ValueError, match="Reserved keyword"):
                validate_cypher_identifier(keyword.lower(), "test")


class TestHTTPSecurity:
    """Test HTTP security middleware functionality"""
    
    def setup_method(self):
        """Set up test app with security middleware"""
        self.app = FastAPI()
        
        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
    
    def test_bearer_token_auth_success(self):
        """Test successful bearer token authentication"""
        expected_token = "test-token-123"
        
        async def auth_middleware(request: Request, call_next):
            auth_header = request.headers.get("Authorization")
            if auth_header != f"Bearer {expected_token}":
                from fastapi import HTTPException, status
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
            return await call_next(request)
        
        self.app.add_middleware(BaseHTTPMiddleware, dispatch=auth_middleware)
        client = TestClient(self.app)
        
        # Test with correct token
        response = client.get("/test", headers={"Authorization": f"Bearer {expected_token}"})
        assert response.status_code == 200
        assert response.json() == {"message": "success"}
    
    def test_bearer_token_auth_failure(self):
        """Test bearer token authentication failure"""
        expected_token = "test-token-123"
        
        async def auth_middleware(request: Request, call_next):
            auth_header = request.headers.get("Authorization")
            if auth_header != f"Bearer {expected_token}":
                from fastapi import HTTPException, status
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
            return await call_next(request)
        
        self.app.add_middleware(BaseHTTPMiddleware, dispatch=auth_middleware)
        client = TestClient(self.app)
        
        # Test without token
        response = client.get("/test")
        assert response.status_code == 401
        
        # Test with wrong token
        response = client.get("/test", headers={"Authorization": "Bearer wrong-token"})
        assert response.status_code == 401
        
        # Test with malformed header
        response = client.get("/test", headers={"Authorization": "Basic wrong-type"})
        assert response.status_code == 401
    
    def test_origin_allowlist_success(self):
        """Test origin allow-list success"""
        allowed_origins = {"https://app.example.com", "https://claude.ai"}
        
        async def origin_middleware(request: Request, call_next):
            origin = (request.headers.get("origin") or "").lower()
            if origin and origin not in allowed_origins:
                from fastapi import HTTPException, status
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Origin not allowed")
            return await call_next(request)
        
        self.app.add_middleware(BaseHTTPMiddleware, dispatch=origin_middleware)
        client = TestClient(self.app)
        
        # Test with allowed origin
        response = client.get("/test", headers={"Origin": "https://claude.ai"})
        assert response.status_code == 200
        
        # Test without origin (should pass)
        response = client.get("/test")
        assert response.status_code == 200
    
    def test_origin_allowlist_failure(self):
        """Test origin allow-list rejection"""
        allowed_origins = {"https://app.example.com", "https://claude.ai"}
        
        async def origin_middleware(request: Request, call_next):
            origin = (request.headers.get("origin") or "").lower()
            if origin and origin not in allowed_origins:
                from fastapi import HTTPException, status
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Origin not allowed")
            return await call_next(request)
        
        self.app.add_middleware(BaseHTTPMiddleware, dispatch=origin_middleware)
        client = TestClient(self.app)
        
        # Test with disallowed origin
        response = client.get("/test", headers={"Origin": "https://malicious.com"})
        assert response.status_code == 403
    
    def test_rate_limiting_setup(self):
        """Test that rate limiting middleware can be configured"""
        # This test mainly ensures the SlowAPI middleware can be set up correctly
        limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])
        self.app.state.limiter = limiter
        self.app.add_middleware(SlowAPIMiddleware)
        
        client = TestClient(self.app)
        
        # First request should succeed
        response = client.get("/test")
        assert response.status_code == 200
        
        # Note: Testing actual rate limit behavior requires more complex test setup
        # with proper request tracking, which is beyond the scope of unit tests 