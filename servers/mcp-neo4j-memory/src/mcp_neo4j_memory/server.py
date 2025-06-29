import os
import logging
import json
from typing import Any, Dict, List, Optional, Literal

import neo4j
from neo4j import GraphDatabase
from pydantic import BaseModel
import asyncio
import typer

import mcp.types as types
from mcp.server.fastmcp import FastMCP

# Set up logging
logger = logging.getLogger('mcp_neo4j_memory')
logger.setLevel(logging.INFO)

# Models for our knowledge graph
class Entity(BaseModel):
    name: str
    type: str
    observations: List[str]

class Relation(BaseModel):
    source: str
    target: str
    relationType: str

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relations: List[Relation]

class ObservationAddition(BaseModel):
    entityName: str
    contents: List[str]

class ObservationDeletion(BaseModel):
    entityName: str
    observations: List[str]

class Neo4jMemory:
    def __init__(self, neo4j_driver):
        self.neo4j_driver = neo4j_driver
        self.create_fulltext_index()

    def create_fulltext_index(self):
        try:
            # TODO , 
            query = """
            CREATE FULLTEXT INDEX search IF NOT EXISTS FOR (m:Memory) ON EACH [m.name, m.type, m.observations];
            """
            self.neo4j_driver.execute_query(query)
            logger.info("Created fulltext search index")
        except neo4j.exceptions.ClientError as e:
            if "An index with this name already exists" in str(e):
                logger.info("Fulltext search index already exists")
            else:
                raise e

    async def load_graph(self, filter_query="*"):
        query = """
            CALL db.index.fulltext.queryNodes('search', $filter) yield node as entity, score
            OPTIONAL MATCH (entity)-[r]-(other)
            RETURN collect(distinct {
                name: entity.name, 
                type: entity.type, 
                observations: entity.observations
            }) as nodes,
            collect(distinct {
                source: startNode(r).name, 
                target: endNode(r).name, 
                relationType: type(r)
            }) as relations
        """
        
        result = self.neo4j_driver.execute_query(query, {"filter": filter_query})
        
        if not result.records:
            return KnowledgeGraph(entities=[], relations=[])
        
        record = result.records[0]
        nodes = record.get('nodes')
        rels = record.get('relations')
        
        entities = [
            Entity(
                name=node.get('name'),
                type=node.get('type'),
                observations=node.get('observations', [])
            )
            for node in nodes if node.get('name')
        ]
        
        relations = [
            Relation(
                source=rel.get('source'),
                target=rel.get('target'),
                relationType=rel.get('relationType')
            )
            for rel in rels if rel.get('source') and rel.get('target') and rel.get('relationType')
        ]
        
        logger.debug(f"Loaded entities: {entities}")
        logger.debug(f"Loaded relations: {relations}")
        
        return KnowledgeGraph(entities=entities, relations=relations)

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        for entity in entities:
            query = f"""
            WITH $entity as entity
            MERGE (e:Memory {{ name: entity.name }})
            SET e += entity {{ .type, .observations }}
            SET e:{entity.type}
            """
            self.neo4j_driver.execute_query(query, {"entity": entity.model_dump()})

        return entities

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        for relation in relations:
            query = f"""
            WITH $relation as relation
            MATCH (from:Memory),(to:Memory)
            WHERE from.name = relation.source
            AND  to.name = relation.target
            MERGE (from)-[r:{relation.relationType}]->(to)
            """
            
            self.neo4j_driver.execute_query(
                query, 
                {"relation": relation.model_dump()}
            )

        return relations

    async def add_observations(self, observations: List[ObservationAddition]) -> List[Dict[str, Any]]:
        query = """
        UNWIND $observations as obs  
        MATCH (e:Memory { name: obs.entityName })
        WITH e, [o in obs.contents WHERE NOT o IN e.observations] as new
        SET e.observations = coalesce(e.observations,[]) + new
        RETURN e.name as name, new
        """
            
        result = self.neo4j_driver.execute_query(
            query, 
            {"observations": [obs.model_dump() for obs in observations]}
        )

        results = [{"entityName": record.get("name"), "addedObservations": record.get("new")} for record in result.records]
        return results

    async def delete_entities(self, entity_names: List[str]) -> None:
        query = """
        UNWIND $entities as name
        MATCH (e:Memory { name: name })
        DETACH DELETE e
        """
        
        self.neo4j_driver.execute_query(query, {"entities": entity_names})

    async def delete_observations(self, deletions: List[ObservationDeletion]) -> None:
        query = """
        UNWIND $deletions as d  
        MATCH (e:Memory { name: d.entityName })
        SET e.observations = [o in coalesce(e.observations,[]) WHERE NOT o IN d.observations]
        """
        self.neo4j_driver.execute_query(
            query, 
            {
                "deletions": [deletion.model_dump() for deletion in deletions]
            }
        )

    async def delete_relations(self, relations: List[Relation]) -> None:
        for relation in relations:
            query = f"""
            WITH $relation as relation
            MATCH (source:Memory)-[r:{relation.relationType}]->(target:Memory)
            WHERE source.name = relation.source
            AND target.name = relation.target
            DELETE r
            """
            self.neo4j_driver.execute_query(
                query, 
                {"relation": relation.model_dump()}
            )

    async def read_graph(self) -> KnowledgeGraph:
        return await self.load_graph()

    async def search_nodes(self, query: str) -> KnowledgeGraph:
        return await self.load_graph(query)

    async def find_nodes(self, names: List[str]) -> KnowledgeGraph:
        return await self.load_graph("name: (" + " ".join(names) + ")")

async def main(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    transport: Literal["stdio", "sse"] = "stdio",
    host: str = "0.0.0.0",
    port: int = 8000,
):
    logger.info(f"Connecting to neo4j MCP Server with DB URL: {neo4j_uri}")

    # Connect to Neo4j
    neo4j_driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_password), 
        database=neo4j_database
    )
    
    # Verify connection
    try:
        neo4j_driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        exit(1)

    # Initialize memory
    memory = Neo4jMemory(neo4j_driver)
    
    # Create MCP server
    server = FastMCP("mcp-neo4j-memory")

    # Register handlers
    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="create_entities",
                description="Create multiple new entities in the knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "The name of the entity"},
                                    "type": {"type": "string", "description": "The type of the entity"},
                                    "observations": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "An array of observation contents associated with the entity"
                                    },
                                },
                                "required": ["name", "type", "observations"],
                            },
                        },
                    },
                    "required": ["entities"],
                },
            ),
            types.Tool(
                name="create_relations",
                description="Create multiple new relations between entities in the knowledge graph. Relations should be in active voice",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "relations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string", "description": "The name of the entity where the relation starts"},
                                    "target": {"type": "string", "description": "The name of the entity where the relation ends"},
                                    "relationType": {"type": "string", "description": "The type of the relation"},
                                },
                                "required": ["source", "target", "relationType"],
                            },
                        },
                    },
                    "required": ["relations"],
                },
            ),
            types.Tool(
                name="add_observations",
                description="Add new observations to existing entities in the knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "observations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entityName": {"type": "string", "description": "The name of the entity to add the observations to"},
                                    "contents": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "An array of observation contents to add"
                                    },
                                },
                                "required": ["entityName", "contents"],
                            },
                        },
                    },
                    "required": ["observations"],
                },
            ),
            types.Tool(
                name="delete_entities",
                description="Delete multiple entities and their associated relations from the knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entityNames": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "An array of entity names to delete"
                        },
                    },
                    "required": ["entityNames"],
                },
            ),
            types.Tool(
                name="delete_observations",
                description="Delete specific observations from entities in the knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "deletions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entityName": {"type": "string", "description": "The name of the entity containing the observations"},
                                    "observations": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "An array of observations to delete"
                                    },
                                },
                                "required": ["entityName", "observations"],
                            },
                        },
                    },
                    "required": ["deletions"],
                },
            ),
            types.Tool(
                name="delete_relations",
                description="Delete multiple relations from the knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "relations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string", "description": "The name of the entity where the relation starts"},
                                    "target": {"type": "string", "description": "The name of the entity where the relation ends"},
                                    "relationType": {"type": "string", "description": "The type of the relation"},
                                },
                                "required": ["source", "target", "relationType"],
                            },
                            "description": "An array of relations to delete"
                        },
                    },
                    "required": ["relations"],
                },
            ),
            types.Tool(
                name="read_graph",
                description="Read the entire knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            types.Tool(
                name="search_nodes",
                description="Search for nodes in the knowledge graph based on a query",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query to match against entity names, types, and observation content"},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="find_nodes",
                description="Find specific nodes in the knowledge graph by their names",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "An array of entity names to retrieve",
                        },
                    },
                    "required": ["names"],
                },
            ),
            types.Tool(
                name="open_nodes",
                description="Open specific nodes in the knowledge graph by their names",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "An array of entity names to retrieve",
                        },
                    },
                    "required": ["names"],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: Dict[str, Any] | None
    ) -> List[types.TextContent | types.ImageContent]:
        try:
            if name == "read_graph":
                result = await memory.read_graph()
                return [types.TextContent(type="text", text=json.dumps(result.model_dump(), indent=2))]

            if not arguments:
                raise ValueError(f"No arguments provided for tool: {name}")

            if name == "create_entities":
                entities = [Entity(**entity) for entity in arguments.get("entities", [])]
                result = await memory.create_entities(entities)
                return [types.TextContent(type="text", text=json.dumps([e.model_dump() for e in result], indent=2))]
                
            elif name == "create_relations":
                relations = [Relation(**relation) for relation in arguments.get("relations", [])]
                result = await memory.create_relations(relations)
                return [types.TextContent(type="text", text=json.dumps([r.model_dump() for r in result], indent=2))]
                
            elif name == "add_observations":
                observations = [ObservationAddition(**obs) for obs in arguments.get("observations", [])]
                result = await memory.add_observations(observations)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "delete_entities":
                await memory.delete_entities(arguments.get("entityNames", []))
                return [types.TextContent(type="text", text="Entities deleted successfully")]
                
            elif name == "delete_observations":
                deletions = [ObservationDeletion(**deletion) for deletion in arguments.get("deletions", [])]
                await memory.delete_observations(deletions)
                return [types.TextContent(type="text", text="Observations deleted successfully")]
                
            elif name == "delete_relations":
                relations = [Relation(**relation) for relation in arguments.get("relations", [])]
                await memory.delete_relations(relations)
                return [types.TextContent(type="text", text="Relations deleted successfully")]
                
            elif name == "search_nodes":
                result = await memory.search_nodes(arguments.get("query", ""))
                return [types.TextContent(type="text", text=json.dumps(result.model_dump(), indent=2))]
                
            elif name == "find_nodes" or name == "open_nodes":
                result = await memory.find_nodes(arguments.get("names", []))
                return [types.TextContent(type="text", text=json.dumps(result.model_dump(), indent=2))]
                
            else:
                raise ValueError(f"Unknown tool: {name}")
                
        except Exception as e:
            logger.error(f"Error handling tool call: {e}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # Start the server using the selected transport
    match transport:
        case "stdio":
            logger.info("MCP Knowledge Graph Memory using Neo4j running on stdio")
            await server.run_stdio_async()
        case "sse":
            logger.info(
                f"MCP Knowledge Graph Memory using Neo4j running via SSE on {host}:{port}"
            )
            await server.run_sse_async(host=host, port=port)
        case _:
            raise ValueError("Transport must be 'stdio' or 'sse'")

app = typer.Typer(help="MCP Neo4j Memory Server")


@app.command()
def cli(
    db_url: str = typer.Option(..., "--db-url", envvar="NEO4J_URL", help="Neo4j Bolt URL"),
    username: str = typer.Option(..., "--username", envvar="NEO4J_USERNAME", help="Neo4j username"),
    password: str = typer.Option(..., "--password", envvar="NEO4J_PASSWORD", help="Neo4j password"),
    database: str = typer.Option("neo4j", "--database", envvar="NEO4J_DATABASE", help="Neo4j database"),
    transport: str = typer.Option("stdio", "--transport", envvar="MCP_TRANSPORT", help="Transport mode: stdio or sse"),
    host: str = typer.Option("0.0.0.0", "--host", envvar="HOST", help="Host for SSE"),
    port: int = typer.Option(8000, "--port", envvar="PORT", help="Port for SSE"),
):
    """Entry point called by the console script."""
    asyncio.run(
        main(
            neo4j_uri=db_url,
            neo4j_user=username,
            neo4j_password=password,
            neo4j_database=database,
            transport=transport,
            host=host,
            port=port,
        )
    )


if __name__ == "__main__":
    app()