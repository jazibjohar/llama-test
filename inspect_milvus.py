from pymilvus import connections, utility, Collection

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# Get all collections
collections = utility.list_collections()
print(f"\nFound {len(collections)} collections:")

# Iterate through each collection
for collection_name in collections:
    print(f"\n{'='*50}")
    print(f"Collection: {collection_name}")
    collection = Collection(collection_name)

    # Print basic info
    print(f"Schema:")
    print(collection.schema)
    print(f"Number of entities: {collection.num_entities}")

    # Query some random vectors
    try:
        results = collection.query(
            expr="", output_fields=["*"], limit=2  # empty expression means "match all"
        )
        print(f"\nSample records (2):", results)
    except Exception as e:
        print(f"Error querying collection: {str(e)}")

connections.disconnect("default")
