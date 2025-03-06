import time
import grpc
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from kafka import KafkaConsumer
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
import faiss
import numpy as np
import os
from collections import OrderedDict
import csv, threading
import concurrent.futures

# Global variables for codes
timestamp = time.strftime("%m%d_%H%M%S")
result_path = f'../europar_results/edgerag_{timestamp}.csv'

class ClusterCache:
    def __init__(self, max_size):        
        self.cache = OrderedDict()  # Store embeddings {cluster_id: {"embeddings": np.array, "latency": float, "count": int}}
        self.max_size = max_size

    def _calculate_eviction_priority(self, cluster_id):
        # Compute eviction priority: (generation latency * visit count)
        # Lower values indicate candidates for eviction.
        return self.cache[cluster_id]["latency"] * self.cache[cluster_id]["count"]

    def evict_if_needed(self, new_cluster_count):
        # Evicts the required number of entries to make space for new clusters.
        # Param new_cluster_count: Number of new clusters that need to be inserted.        
        current_size = len(self.cache)
        if current_size + new_cluster_count > self.max_size:
            # Determine how many entries need to be evicted
            num_to_evict = (current_size + new_cluster_count) - self.max_size
            print(f"[Cache Eviction] Removing {num_to_evict} entries.")

            # Sort clusters by eviction priority (ascending order)            
            eviction_candidates = sorted(self.cache.keys(), key=self._calculate_eviction_priority)[:num_to_evict]

            # Remove the lowest priority entries
            for cluster_id in eviction_candidates:
                del self.cache[cluster_id]
                print(f"[Cache Eviction] Evicted Cluster {cluster_id}")

    def get(self, cluster_id):        
        if cluster_id in self.cache:
            self.cache[cluster_id]["count"] += 1  # Increase visit count
            self.cache.move_to_end(cluster_id)  # Mark as recently used
            return self.cache[cluster_id]["embeddings"]
        return None

    def put(self, cluster_id, embeddings, generation_latency):
        """Insert a new cluster into the cache, applying eviction policy if necessary."""
        if cluster_id in self.cache:
            # Update existing entry
            self.cache[cluster_id]["embeddings"] = embeddings
            self.cache[cluster_id]["latency"] = generation_latency
            self.cache[cluster_id]["count"] += 1
        else:
            if len(self.cache) >= self.max_size:
                # Evict the lowest-priority entry
                eviction_target = min(self.cache, key=self._calculate_eviction_priority)
                #print(f"[Cache Eviction] Removing Cluster {eviction_target}")
                del self.cache[eviction_target]

            # Insert the new cluster
            self.cache[cluster_id] = {"embeddings": embeddings, "latency": generation_latency, "count": 1}
            self.cache.move_to_end(cluster_id)  # Mark as recently used

class EdgeRAGWithCache:
    def __init__(self, ivf_centroids_path, faiss_index, cache_size, cache_miss_threshold=0.3):
        # Coarse_Quantizer
        self.faiss_index = faiss_index
        self.coarse_quantizer = faiss.read_index(ivf_centroids_path)

        self.cache = ClusterCache(max_size=cache_size)
        self.cluster_generation_latency = {}  # Store precomputed latencies

        # Track for cache hit ratios
        self.total_cluster_requests = 0
        self.total_cache_hits = 0
        self.idx_cnt = 1
        self.warmupCnt = 1 # First msgs are used for warmup(1: warum-up, 2: experiment start)

        # Sort for queries with similiar cluster ids
        self.query_buffer = []  # Temporary query storage for sorting

        # Detect whether prefetch entries are needed for switching of cluster groups
        self.prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self.prefetch_future = None  # Track prefetch task

        # Prefetch within same cluster group
        self.cache_miss_threshold = cache_miss_threshold  # Prefetch when cache miss rate exceeds threshold

        # create csv file to store experiment results        
        with open(result_path, mode='w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Seq", "Duration", "EBT", "FLT", "CLT", "SLT", "MIT", "VST", "CHR"])
    
    def write_results_to_csv(self, seq, tt, ebt, flt, clt, slt, mit, vst, chr):
        with open(result_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([seq, tt, ebt, flt, clt, slt, mit, vst, chr])

    def prefetch_cluster(self, next_query, next_cluster_ids):
        self.cache.evict_if_needed(len(next_cluster_ids))

        for cluster_id in next_cluster_ids:
            if cluster_id not in self.cache.cache:
                converted_cluster_id = int(str(cluster_id))
                new_embeddings = np.load(f"./disk_clusters/cluster_{cluster_id}.npy")
                generation_latency = self.cluster_generation_latency.get(converted_cluster_id, float("inf"))
                self.cache.put(cluster_id, new_embeddings, generation_latency)
                # converted_cluster_id = int(str(cluster_id))
                # list_size = self.faiss_index.invlists.list_size(converted_cluster_id)
                # if list_size == 0:
                #     continue
                # vector_ids_ptr = self.faiss_index.invlists.get_ids(converted_cluster_id)
                # vector_ids = faiss.rev_swig_ptr(vector_ids_ptr, list_size).astype(np.int64)
                # new_embeddings = np.array([self.faiss_index.reconstruct(int(vector_id)) for vector_id in vector_ids])
                # generation_latency = self.cluster_generation_latency.get(converted_cluster_id, float("inf"))
                # self.cache.put(cluster_id, new_embeddings, generation_latency)
        print(f"[Prefetching Query: {next_query}] Completed for clusters:", next_cluster_ids)

    def _profile_cluster_generation(self):
        for cluster_id in range(self.faiss_index.nlist):
            start_time = time.time()

            list_size = self.faiss_index.invlists.list_size(cluster_id)
            
            if list_size == 0:
                self.cluster_generation_latency[cluster_id] = float("inf")
                continue

            vector_ids_ptr = self.faiss_index.invlists.get_ids(cluster_id)
            vector_ids = faiss.rev_swig_ptr(vector_ids_ptr, list_size).astype(np.int64)

            # Generate embeddings (but do not store them)
            _ = [self.faiss_index.reconstruct(int(vector_id)) for vector_id in vector_ids]

            # Store measured latency
            self.cluster_generation_latency[cluster_id] = time.time() - start_time

        print("[Precompute] Cluster generation latencies computed successfully.")

    def predict_cache_miss_rate(self, cluster_ids):
        """Predict the cache miss rate for the upcoming query."""
        cache_misses = sum(1 for cluster_id in cluster_ids if cluster_id not in self.cache.cache)
        return cache_misses / len(cluster_ids) if cluster_ids else 0      

    def search(self, query_text, model, prefetched_dict, extracted_all_data, k=10):
        if self.prefetch_future is not None:
            self.prefetch_future.result()

        encodeing_start_time_at_search = time.time()
        query_vector = model.encode([query_text]) #, convert_to_numpy=True)[0]
        encodeing_end_time_at_search = time.time() - encodeing_start_time_at_search

        firstlookup_start_time_at_search = time.time()
        _, cluster_ids = self.coarse_quantizer.search(query_vector, 10)
        firstlookup_end_time_at_search = time.time() - firstlookup_start_time_at_search
        cluster_ids = cluster_ids[0]  # Extract cluster IDs, return numpy.int64, so it has to convert to int

        # Track total cluster requests(Evaluation)
        self.total_cluster_requests += len(cluster_ids)
        cache_hits = 0  # Track hits per query
        cache_miss = 0
        
        temp_embeddings = [] # embeddings for selective search
        missing_clusters = [] # need for online generation

        # At cache-level search, it first lookups cache.
        # If the embeddings are found in the cache, hit!. Otherwise, missing_clusters are recorded
        cachelookup_start_time_at_search = time.time()
        for cluster_id in cluster_ids:
            cached_embeddings = self.cache.get(cluster_id)
            
            if cached_embeddings is not None:
                #print(f"[Cache Hit] Using Precomputed Cluster {cluster_id}")
                temp_embeddings.append(cached_embeddings)
                cache_hits += 1
            else:
                #print(f"[Cache Miss] Cluster {cluster_id} needs embedding generation.")
                missing_clusters.append(cluster_id)
                cache_miss += 1
        cachelookup_end_time_at_search = time.time() - cachelookup_start_time_at_search

        # Update total cache hits
        self.total_cache_hits += cache_hits

        # If missing_clusters are existed, cache replacement is performed.
        secondlookup_start_time_at_search = time.time()
        if missing_clusters is not None:
            self.cache.evict_if_needed(len(missing_clusters)) # Evict multiple entries if needed before inserting new ones
            
            for cluster_id in missing_clusters: # Put new entries to the cache, I think this online generation phase should be processed in parallel
                index = np.load(f"./disk_clusters/cluster_{cluster_id}.npy")
        
                # Use precomputed latency
                generation_latency = self.cluster_generation_latency.get(cluster_id, float("inf"))
                self.cache.put(cluster_id, index, generation_latency)
                temp_embeddings.append(index)
        secondlookup_end_time_at_search = time.time() - secondlookup_start_time_at_search

        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB 단위
        print(f"memory usage: {memory_usage} MB")

        mergeindex_start_time_at_search = time.time()
        merged_embeddings = np.vstack(temp_embeddings) #
        temp_index = faiss.IndexFlatL2(merged_embeddings.shape[1])
        temp_index.add(merged_embeddings) # much time.
        mergeindex_end_time_at_search = time.time() - mergeindex_start_time_at_search

        vectorsearch_start_time_at_search = time.time()
        D, I = temp_index.search(query_vector, k) # query_vector.reshape(1, -1)
        vectorsearch_end_time_at_search = time.time() - vectorsearch_start_time_at_search
        total_duration = time.time() - encodeing_start_time_at_search

        # After vector search, check whether cluster group will be switched
        keys = list(sorted(prefetched_dict.keys()))

        for key in prefetched_dict:
            current_index = keys.index(key)
            if current_index + 1 < len(keys): # avoid -- list index out of range
                nxt_key = keys[current_index+1]
            if query_text == prefetched_dict[key]['LQ']: # last query of current cluster group -> This is timing to prefetch because cluster group will be switched
                first_query_next_cluster = prefetched_dict[nxt_key]['FQ'] # first query of next cluster group
                #print(f"[Last Query of group]: {prefetched_dict[key]['LQ']} //// [Current Query Text: {query_text} /// [First Query of next group] {first_query_next_cluster}")
                first_query_cluster_ids_next_cluster = prefetched_dict[nxt_key]['FQSET'] # first query's clusterIDs of next cluster group
                self.prefetch_future = self.prefetch_executor.submit(self.prefetch_cluster, first_query_next_cluster, first_query_cluster_ids_next_cluster)
        
        # After vector search, check whether next query in same cluster group will be prefetched
        missing_clusters_next_query = [] # need for online generation
        for key, data in extracted_all_data.items():
            queries = data["queries"]
            frozensets = data["frozensets"]

            if query_text in queries:
                index = queries.index(query_text)

                if index < len(queries)-1:
                    next_query = queries[index+1]
                    next_frozensets = frozensets[index+1]

                    for cluster_id in next_frozensets: # compute next query's cache miss rate
                        cached_embeddings = self.cache.get(cluster_id)
            
                        if cached_embeddings is not None:
                            print(f"[Next Query Cache Hit] Using Precomputed Cluster {cluster_id}")
                        else:
                            print(f"[Next Query Cache Miss] Cluster {cluster_id} needs embedding generation.")
                            missing_clusters_next_query.append(cluster_id)

        if len(missing_clusters_next_query) > 2:
            self.prefetch_future = self.prefetch_executor.submit(self.prefetch_cluster, next_query, next_frozensets)
        
        # Compute cache hit ratio for this query
        total_cache_num = cache_hits + cache_miss
        cache_hit_ratio = cache_hits / total_cache_num

        if self.warmupCnt != 1:
            #print(f"TT: {total_duration}, EBT: {encodeing_end_time_at_search:.4f}, FLT: {firstlookup_end_time_at_search:.4f}, CLT: {cachelookup_end_time_at_search:.4f}, SLT: {secondlookup_end_time_at_search:.4f}, MIT: {mergeindex_end_time_at_search:.4f}, VST: {vectorsearch_end_time_at_search:.4f}, CHR: {cache_hit_ratio:.2%}")
            self.write_results_to_csv(f"{self.idx_cnt}",
                                  f"{total_duration:.3f}",
                                  f"{encodeing_end_time_at_search:.3f}",
                                  f"{firstlookup_end_time_at_search:.3f}",
                                  f"{cachelookup_end_time_at_search:.3f}",
                                  f"{secondlookup_end_time_at_search:.3f}",
                                  f"{mergeindex_end_time_at_search:.3f}",
                                  f"{vectorsearch_end_time_at_search:.3f}",
                                  f"{cache_hit_ratio:.3%}"
                                  )
            self.idx_cnt += 1
        self.warmupCnt += 1

        return I[0], D[0]
    
    def get_total_cache_hit_ratio(self):
        if self.total_cluster_requests == 0:
            return 0
        return self.total_cache_hits / self.total_cluster_requests # overall cache utilization
    
##### Jaccard, clustering algorithm #####

def compute_jaccard_similarity(set1, set2):
        if not set1 or not set2:
            return 0  # No overlap
        return len(set1 & set2) / len(set1 | set2)

def extract_all_queries_and_frozensets(cluster_groups):
    """
    Extract all queries and frozensets per key from the given dictionary.
    Stores them in a new dictionary.
    """
    extracted_data = {}

    for key, value in cluster_groups.items():
        # Extract all queries and frozensets from the list
        queries = [query_text for query_text, _ in value]
        frozensets = [query_set for _, query_set in value]

        # Store them in a dictionary
        extracted_data[key] = {
            "queries": queries,
            "frozensets": frozensets
        }

    return extracted_data

def sort_queries_by_clustering(edgeRagSearcher, messages, model):
    cluster_to_queries = {}
    query_vectors = []
    query_texts = []
    cluster_sets = []
    cluster_id_mapping = {}  # Store cluster IDs per query

    # 1: Extract cluster IDs for each query
    for query_text in messages:
        query_vector = model.encode([query_text])
        _, cluster_ids = edgeRagSearcher.coarse_quantizer.search(query_vector, 10)
        cluster_ids_set = frozenset(cluster_ids[0]) 

        cluster_to_queries[query_text] = (query_vector, cluster_ids_set)
        query_vectors.append(query_vector)
        query_texts.append(query_text)
        cluster_sets.append(cluster_ids_set)
        cluster_id_mapping[query_text] = cluster_ids_set

    num_queries = len(query_texts)

    # Step 2: Compute Jaccard distance matrix (1 - similarity)
    jaccard_matrix = np.zeros((num_queries, num_queries))
    
    for i in range(num_queries):
        for j in range(num_queries):
            if i != j:
                jaccard_matrix[i, j] = compute_jaccard_similarity(cluster_sets[i], cluster_sets[j])
    
    # Convert to distance matrix (1 - similarity)
    jaccard_distance_matrix = 1 - jaccard_matrix

    # Step 3: Perform Agglomerative Clustering    
    optimal_distance_threshold = 0.7
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=optimal_distance_threshold, metric="precomputed", linkage="average")
    cluster_labels = clustering.fit_predict(jaccard_distance_matrix)
    
    # Debugging for group queries by their assigned clusters
    cluster_groups = {}

    for query, assigned_cluster_id in zip(query_texts, cluster_labels):
        if assigned_cluster_id not in cluster_groups:
            cluster_groups[assigned_cluster_id] = []

        cluster_groups[assigned_cluster_id].append((query, cluster_id_mapping[query]))  # Store FAISS cluster IDs
    
    extracted_all_data = extract_all_queries_and_frozensets(cluster_groups)
    # print("#################")
    # for key, value in cluster_groups.items():
    #     for query, id in value:
    #         print(f"[Query]: {query}, Frozenset: {id}")
    # print("#################")
    # Step 4. Extract only the first and last query of each group ID for prefetch    
    #first_last_queries_per_cluster = { cluster_id: (queries[0][0], queries[0][1], queries[-1][0,], queries[-1][1]) for cluster_id, queries in cluster_groups.items() }
    prefetched_dict = {
        key: {
            "FQ": value[0][0],
            "FQSET": value[0][1],
            "LQ": value[-1][0],
            "LQSET": value[-1][1],
        }
        for key, value in cluster_groups.items()
    }

    # current_query = "What happens to Van Martin after Jeff accidentally knocks the gun from his hands?"
    # next_query, next_frozenset, next_cluster_id = get_next_query_info(cluster_groups, current_query)

    # print(f"Next Query: {next_query}")
    # print(f"Next Query's Frozenset: {next_frozenset}")
    # print(f"Next Query's Cluster ID: {next_cluster_id}")
    # print("##################")
    # keys = list(extracted_data.keys())
    # print(*keys)
    # for key in extracted_data:
    #     current_index = keys.index(key)
    #     if current_index + 1 < len(keys):
    #         nxt_key = keys[current_index+1]
    #     # print(f"current key: {key} and next key: {nxt_key}")
    #     if "What will cause Fanshaw to die young?" == extracted_data[key]['LQ']:
    #         if current_index + 1 < len(keys):
    #             first_query_next_cluster = extracted_data[nxt_key]['FQ']
    #             first_query_cluster_ids_next_cluster = extracted_data[nxt_key]['FQSET']
    #             print(f"FQ: {first_query_next_cluster}, FQSET: {first_query_cluster_ids_next_cluster}")    
    # print("##################")
    
    # Step 5: Sort queries by cluster assignments
    sorted_queries = sorted(zip(cluster_labels, query_texts, query_vectors), key=lambda x: x[0])

    return [(q, v) for _, q, v in sorted_queries], prefetched_dict, extracted_all_data

def kafka_search(centroid_path):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    edgeRagSearcher = EdgeRAGWithCache(centroid_path, 40)
    edgeRagSearcher._profile_cluster_generation() # Pre-compute cluster generation() before search phase

    consumer = KafkaConsumer(
        "hotpotqa",
        bootstrap_servers="163.239.199.205:9092",
        auto_offset_reset='earliest',  # 가장 처음부터 읽기 ('latest'로 설정하면 최신 메시지만 읽음)
        enable_auto_commit=True,  # 자동 커밋 활성화
        value_deserializer=lambda x: x.decode('utf-8')  # 바이트 데이터를 문자열로 변환
    )

    messages = []  # 메시지를 담을 리스트

    for message in consumer:
        messages.append(message.value)

        # 리스트에 10개가 쌓이거나 1초가 지나면 출력 후 초기화
        if len(messages) >= 100:
            print("Received message count:", len(messages))

            # Before vector search, sorting should be performed
            sorted_time = time.time()
            sorted_queries, prefetched_dict, extracted_all_data = sort_queries_by_clustering(edgeRagSearcher, messages, model)
            sorted_duration = time.time() - sorted_time
            print(f"sorted time: {sorted_duration}")

            for query_text, query_vector in sorted_queries:
                print(f"******** Current Query is {query_text}")
                edgeRagSearcher.search(query_text, model, prefetched_dict, extracted_all_data)

            messages.clear()
            time.sleep(2)
        # overall_hit_ratio = edgeRagSearcher.get_total_cache_hit_ratio()
        # print(f"[Overall Cache Hit Ratio] {overall_hit_ratio:.2%}")

if __name__ == "__main__":
    inf_centroids_file ="../hotpotqa_centroids.index"
    kafka_search(inf_centroids_file)