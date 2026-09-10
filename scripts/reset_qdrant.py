from config.vector_dependency import vector_store

print("Resetting Qdrant collection...")

vector_store.reset_collection()

print("Qdrant collection reset complete.")