"""
Quick start example for SynapCores Python SDK.
"""

import asyncio
from synapcores import SynapCores


async def main():
    # Initialize client
    client = SynapCores(
        host="localhost",
        port=8080,
        api_key="your-api-key"
    )
    
    # Create a collection
    products = client.create_collection(
        name="products",
        schema={
            "name": "string",
            "description": "text",
            "price": "float",
            "category": "string",
            "embedding": "vector[384]"
        }
    )
    
    # Insert some products
    products.insert([
        {
            "name": "Wireless Mouse",
            "description": "Ergonomic wireless mouse with precision tracking",
            "price": 29.99,
            "category": "Electronics"
        },
        {
            "name": "Mechanical Keyboard",
            "description": "RGB mechanical keyboard with Cherry MX switches",
            "price": 129.99,
            "category": "Electronics"
        },
        {
            "name": "USB-C Hub",
            "description": "7-in-1 USB-C hub with HDMI, USB 3.0, and SD card reader",
            "price": 49.99,
            "category": "Electronics"
        }
    ])
    
    # Semantic search
    print("Searching for computer peripherals...")
    results = products.search(
        query="computer peripherals for gaming",
        top_k=5
    )
    
    for doc in results.documents:
        print(f"- {doc.data['name']} (${doc.data['price']}) - Score: {doc.score:.3f}")
    
    # SQL query with AI extensions
    print("\nRunning SQL query with embeddings...")
    df = client.sql("""
        SELECT name, price,
               similarity(embedding, embed('gaming accessories')) as relevance
        FROM products
        WHERE category = 'Electronics' AND price < 100
        ORDER BY relevance DESC
    """)
    print(df)
    
    # NLP analysis
    print("\nAnalyzing product reviews...")
    review = "This keyboard is amazing! The RGB lighting is beautiful and the switches feel great."
    analysis = client.nlp.analyze(review, tasks=["sentiment", "keywords"])
    
    print(f"Sentiment: {analysis.sentiment.label} ({analysis.sentiment.score:.3f})")
    print(f"Keywords: {', '.join(analysis.keywords)}")
    
    # Real-time updates (async)
    print("\nSubscribing to price changes...")
    
    def handle_change(event):
        print(f"Price updated: {event.document.data['name']} - ${event.document.data['price']}")
    
    subscription = await products.subscribe(
        filter={"price": {"$lt": 50}},
        on_change=handle_change
    )
    
    # Keep subscription alive for demo
    await asyncio.sleep(10)
    await subscription.close()
    
    client.close()


if __name__ == "__main__":
    asyncio.run(main())