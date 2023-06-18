#https://www.programcreek.com/python/example/112291/faiss.GpuIndexFlatL2
def _retrieve_knn_faiss_gpu_euclidean(query_embeddings, db_embeddings, k, gpu_id=0):
    """
        Retrieve k nearest neighbor based on inner product

        Args:
            query_embeddings:           numpy array of size [NUM_QUERY_IMAGES x EMBED_SIZE]
            db_embeddings:              numpy array of size [NUM_DB_IMAGES x EMBED_SIZE]
            k:                          number of nn results to retrieve excluding query
            gpu_id:                     gpu device id to use for nearest neighbor (if possible for `metric` chosen)

        Returns:
            dists:                      numpy array of size [NUM_QUERY_IMAGES x k], distances of k nearest neighbors
                                        for each query
            retrieved_db_indices:       numpy array of size [NUM_QUERY_IMAGES x k], indices of k nearest neighbors
                                        for each query
    """
    import faiss

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = gpu_id

    # Evaluate with inner product
    index = faiss.GpuIndexFlatL2(res, db_embeddings.shape[1], flat_config)
    index.add(db_embeddings)
    # retrieved k+1 results in case that query images are also in the db
    dists, retrieved_result_indices = index.search(query_embeddings, k + 1)

    return dists, retrieved_result_indices 
