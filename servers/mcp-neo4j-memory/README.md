# 🧠🕸️ Neo4j Knowledge Graph Memory MCP Server

## 🌟 Overview

A Model Context Protocol (MCP) server implementation that provides persistent memory capabilities through Neo4j graph database integration.

By storing information in a graph structure, this server maintains complex relationships between entities as memory nodes and enables long-term retention of knowledge that can be queried and analyzed across multiple conversations or sessions.

With [Neo4j Aura](https://console.neo4j.io) you can host your own database server for free or share it with your collaborators. Otherwise you can run your own Neo4j server locally.

The MCP server leverages Neo4j's graph database capabilities to create an interconnected knowledge base that serves as an external memory system. Through Cypher queries, it allows exploration and retrieval of stored information, relationship analysis between different data points, and generation of insights from the accumulated knowledge. This memory can be further enhanced with Claude's capabilities.

### 🕸️ Graph Schema

* `Memory` - A node representing an entity with a name, type, and observations.
* `Relationship` - A relationship between two entities with a type.

### 🔍 Usage Example

```
Let's add some memories 
I, Michael, living in Dresden, Germany work at Neo4j which is headquartered in Sweden with my colleagues Andreas (Cambridge, UK) and Oskar (Gothenburg, Sweden)
I work in Product Management, Oskar in Engineering and Andreas in Developer Relations.
```

Results in Claude calling the create_entities and create_relations tools.

![](./docs/images/employee_create_entities_and_relations.png)

![](./docs/images/employee_graph.png)

## 📦 Components

### 🔧 Tools

The server offers these core tools:

#### 🔎 Query Tools
- `read_graph`
   - Read the entire knowledge graph
   - No input required
   - Returns: Complete graph with entities and relations

- `search_nodes`
   - Search for nodes based on a query
   - Input:
     - `query` (string): Search query matching names, types, observations
   - Returns: Matching subgraph

- `find_nodes`
   - Find specific nodes by name
   - Input:
     - `names` (array of strings): Entity names to retrieve
   - Returns: Subgraph with specified nodes

#### ♟️ Entity Management Tools
- `create_entities`
   - Create multiple new entities in the knowledge graph
   - Input:
     - `entities`: Array of objects with:
       - `name` (string): Name of the entity
       - `type` (string): Type of the entity  
       - `observations` (array of strings): Initial observations about the entity
   - Returns: Created entities

- `delete_entities` 
   - Delete multiple entities and their associated relations
   - Input:
     - `entityNames` (array of strings): Names of entities to delete
   - Returns: Success confirmation

#### 🔗 Relation Management Tools
- `create_relations`
   - Create multiple new relations between entities
   - Input:
     - `relations`: Array of objects with:
       - `source` (string): Name of source entity
       - `target` (string): Name of target entity
       - `relationType` (string): Type of relation
   - Returns: Created relations

- `delete_relations`
   - Delete multiple relations from the graph
   - Input:
     - `relations`: Array of objects with same schema as create_relations
   - Returns: Success confirmation

#### 📝 Observation Management Tools
- `add_observations`
   - Add new observations to existing entities
   - Input:
     - `observations`: Array of objects with:
       - `entityName` (string): Entity to add to
       - `contents` (array of strings): Observations to add
   - Returns: Added observation details

- `delete_observations`
   - Delete specific observations from entities
   - Input:
     - `deletions`: Array of objects with:
       - `entityName` (string): Entity to delete from
       - `observations` (array of strings): Observations to remove
   - Returns: Success confirmation

## 🔧 Usage with Claude Desktop

### 💾 Installation

```bash
pip install mcp-neo4j-memory
```

### ⚙️ Configuration

Add the server to your `claude_desktop_config.json` with configuration of:

```json
"mcpServers": {
  "neo4j": {
    "command": "uvx",
    "args": [
      "mcp-neo4j-memory@0.1.5",
      "--db-url",
      "neo4j+s://xxxx.databases.neo4j.io",
      "--username",
      "<your-username>",
      "--password",
      "<your-password>"
    ]
  }
}
```

Alternatively, you can set environment variables:

```json
"mcpServers": {
  "neo4j": {
    "command": "uvx",
    "args": [ "mcp-neo4j-memory@0.1.5" ],
    "env": {
      "NEO4J_URL": "neo4j+s://xxxx.databases.neo4j.io",
      "NEO4J_USERNAME": "<your-username>",
      "NEO4J_PASSWORD": "<your-password>"
    }
  }
}
```

### 🐳 Using with Docker

```json
"mcpServers": {
  "neo4j": {
    "command": "docker",
    "args": [
      "run",
      "--rm",
      "-e", "NEO4J_URL=neo4j+s://xxxx.databases.neo4j.io",
      "-e", "NEO4J_USERNAME=<your-username>",
      "-e", "NEO4J_PASSWORD=<your-password>",
      "mcp/neo4j-memory:0.1.5"
    ]
  }
}
```

## 🚀 Development

### 📦 Prerequisites

1. Install `uv` (Universal Virtualenv):
```bash
# Using pip
pip install uv

# Using Homebrew on macOS
brew install uv

# Using cargo (Rust package manager)
cargo install uv
```

2. Clone the repository and set up development environment:
```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-neo4j-memory.git
cd mcp-neo4j-memory

# Create and activate virtual environment using uv
uv venv
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows

# Install dependencies including dev dependencies
uv pip install -e ".[dev]"
```

### 🐳 Docker

Build and run the Docker container:

```bash
# Build the image
docker build -t mcp/neo4j-memory:latest .

# Run the container
docker run -e NEO4J_URL="neo4j+s://xxxx.databases.neo4j.io" \
          -e NEO4J_USERNAME="your-username" \
          -e NEO4J_PASSWORD="your-password" \
          mcp/neo4j-memory:latest
```

## 📄 License

This MCP server is licensed under the MIT License. This means you are free to use, modify, and distribute the software, subject to the terms and conditions of the MIT License. For more details, please see the LICENSE file in the project repository.

## 🔒 Authentication & Security

Streamable-HTTP deployments **should always be protected** with a bearer token.  The server refuses requests whose header is not:

```http
Authorization: Bearer <YOUR_TOKEN>
```

### Choosing a token
• Use at least **128 bits of entropy** (16 random bytes ⇒ 32 hex chars).
• 256-bit tokens (64 hex chars) give more margin and are still short.
• Avoid dictionary words.  Generate with one of:

```bash
# 128-bit hex
openssl rand -hex 16            # 32-char token

# 256-bit hex
openssl rand -hex 32            # 64-char token

# URL-safe base64
python - <<'PY'
import secrets, base64, os
print(base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b'=')).decode()
PY
```

### Enabling the token in the server
Either pass the CLI flag or set the env-var:

```bash
export MCP_TOKEN="<YOUR_TOKEN>"

mcp-neo4j-memory \
  --transport http \
  --token $MCP_TOKEN \
  --host 0.0.0.0 --port 8000
```

### Origin allow-list (optional)
Limit browser requests to specific origins:

```bash
export MCP_ALLOW_ORIGINS="https://app.example,https://claude.ai"
```

### Rate limiting (optional)
Protect against brute-force and abuse:

```bash
export MCP_RATE_LIMIT="200/minute"   # SlowAPI syntax
```

### Using with Anthropic's MCP connector
Configure the server object in your `mcp_servers` array:

```json
{
  "type": "url",
  "url": "https://neo4j.example.com/mcp",   // must be HTTPS
  "name": "neo4j-memory",
  "authorization_token": "<YOUR_TOKEN>"
}
```

Claude will include the bearer token on every request and pass the server's security checks.  See Anthropic docs on the MCP connector for details.  

## 🌐 Exposing the server over HTTPS (production)

FastMCP's built-in web runner is **HTTP-only**.  For production you terminate TLS in a reverse proxy that forwards requests to the container's internal port (`8000`).

Below are turnkey recipes for the two most common choices.  In both cases make sure you forward the `Authorization` header so the bearer-token check works.

### Option 1 — Nginx (Let's Encrypt / Certbot)

1  Install Nginx & Certbot (or use the official docker image + `nginx-proxy-manager`).

2  Create an upstream block pointing to the container:

```nginx
upstream mcp_neo4j_memory {
    server 127.0.0.1:8000;   # container published port
    keepalive 32;
}

server {
    listen 443 ssl;
    server_name neo4j.example.com;

    ssl_certificate     /etc/letsencrypt/live/neo4j.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/neo4j.example.com/privkey.pem;

    # Forward real client IP so rate-limiter sees it
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    location /mcp {
        proxy_pass http://mcp_neo4j_memory;
        proxy_set_header Host $host;
        proxy_pass_request_headers on;           # keep Authorization header
        proxy_http_version 1.1;                  # keeps SSE / streaming happy
        proxy_buffering off;                     # stream responses immediately
    }
}
```

3  Reload Nginx and confirm `https://neo4j.example.com/mcp` returns the MCP `Initialize` result when called with your bearer header.

#### Using Certbot to obtain / renew the cert

If you already have Nginx running on the host, the easiest flow is the **Certbot Nginx plugin** which edits the config and sets up renewal hooks automatically:

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install certbot python3-certbot-nginx

# Obtain a certificate for the MCP domain (interactive prompts)
sudo certbot --nginx -d neo4j.example.com

# Dry-run renewal once to verify timers/hooks work
sudo certbot renew --dry-run
```

Certbot will write the certificate and key to:
```
+/etc/letsencrypt/live/neo4j.example.com/fullchain.pem
+/etc/letsencrypt/live/neo4j.example.com/privkey.pem
```
and insert an `include` block or `ssl_certificate` lines into your Nginx site file.  The sample Nginx snippet above already references those paths.

Certificates renew automatically (twice-daily systemd timer); Nginx is reloaded by Certbot after each renewal so no manual action is required.

### Option 2 — Caddy (automatic certificates)

```caddyfile
neo4j.example.com {
    encode gzip
    reverse_proxy /mcp* 127.0.0.1:8000 {
        header_up Authorization {>Authorization}
        header_up X-Forwarded-For {remote_host}
    }
}
```

Caddy will obtain and renew TLS certs automatically via Let's Encrypt.

### Docker-Compose example (app + Caddy)

```yaml
version: "3.9"

services:
  memory:
    image: mcp-neo4j-memory:latest      # build from repo
    environment:
      - MCP_TOKEN=${MCP_TOKEN}
      - NEO4J_URL=${NEO4J_URL}
      - NEO4J_USERNAME=${NEO4J_USERNAME}
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
    command: >
      mcp-neo4j-memory --transport http --host 0.0.0.0 --port 8000

  caddy:
    image: caddy:2-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy_data:/data

volumes:
  caddy_data:
```

Once the stack is up, visit `https://neo4j.example.com/mcp` with the bearer token header to verify.

### Updating the Anthropic configuration

Use the HTTPS URL in the `mcp_servers` array as previously shown—no further changes needed.

---
