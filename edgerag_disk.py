import time
import grpc
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from kafka import KafkaConsumer
import faiss
import numpy as np
import os
from collections import OrderedDict
import csv, psutil, gc

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
                gc.collect()
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
                print(f"[Cache Eviction] Removing Cluster {eviction_target}")
                del self.cache[eviction_target]

            # Insert the new cluster
            self.cache[cluster_id] = {"embeddings": embeddings, "latency": generation_latency, "count": 1}
            self.cache.move_to_end(cluster_id)  # Mark as recently used

class EdgeRAGWithCache:
    def __init__(self, ivf_centroids_path, cache_size):
        # Coarse_Quantizer        
        self.coarse_quantizer = faiss.read_index(ivf_centroids_path)

        self.cache = ClusterCache(max_size=cache_size)
        self.cluster_generation_latency = {}  # Store precomputed latencies

        # Track for cache hit ratios
        self.total_cluster_requests = 0
        self.total_cache_hits = 0
        self.idx_cnt = 1
        self.warmupCnt = 1 # First msgs are used for warmup(1: warum-up, 2: experiment start)

        # create csv file to store experiment results        
        with open(result_path, mode='w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Seq", "Duration", "EBT", "FLT", "CLT", "SLT", "MIT", "VST", "CHR"])
    
    def write_results_to_csv(self, seq, tt, ebt, flt, clt, slt, mit, vst, chr):
        with open(result_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([seq, tt, ebt, flt, clt, slt, mit, vst, chr])

    def _profile_cluster_generation(self):
        for cluster_id in range(0,100):
            start_time = time.time()

            # list_size = self.faiss_index.invlists.list_size(cluster_id)
            
            # if list_size == 0:
            #     self.cluster_generation_latency[cluster_id] = float("inf")
            #     continue

            # vector_ids_ptr = self.faiss_index.invlists.get_ids(cluster_id)
            # vector_ids = faiss.rev_swig_ptr(vector_ids_ptr, list_size).astype(np.int64)

            # Generate embeddings (but do not store them)
            #print(f"embeddings_per_cluster is performing")
            # embeddings_per_cluster = [self.faiss_index.reconstruct(int(vector_id)) for vector_id in vector_ids]
            np.load(f"./disk_clusters/cluster_{cluster_id}.npy")
            #faiss.write_index(embeddings_per_cluster, f"disk_clusters/cluster_{cluster_id}.faiss")
            #np.save(f"disk_clusters/cluster_{cluster_id}.npy", embeddings_per_cluster)

            # Store measured latency
            self.cluster_generation_latency[cluster_id] = time.time() - start_time

        print("[Precompute] Cluster generation latencies computed successfully.")        

    def search(self, query_text, model, k=10):
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
                print(f"[Cache Hit] Using Precomputed Cluster {cluster_id}")
                temp_embeddings.append(cached_embeddings)
                cache_hits += 1
            else:
                print(f"[Cache Miss] Cluster {cluster_id} needs embedding generation.")
                missing_clusters.append(cluster_id)
                cache_miss += 1
        cachelookup_end_time_at_search = time.time() - cachelookup_start_time_at_search

        # Update total cache hits
        self.total_cache_hits += cache_hits

        # If missing_clusters are existed, cache replacement is performed.
        merged_index = faiss.IndexFlatL2(768)
        secondlookup_start_time_at_search = time.time()
        if missing_clusters is not None:
            self.cache.evict_if_needed(len(missing_clusters)) # Evict multiple entries if needed before inserting new ones
            
            for cluster_id in missing_clusters: # Put new entries to the cache, I think this online generation phase should be processed in parallel                
                index = np.load(f"./disk_clusters/cluster_{cluster_id}.npy") # index = faiss.read_index(f"disk_clusters/cluster_{cluster_id}.npy")
                
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
        print(f"[Message]:", message.value)

        # 리스트에 10개가 쌓이거나 1초가 지나면 출력 후 초기화
        if len(messages) >= 100:
            #print("Received message count:", len(messages))

            for msg in messages:
                edgeRagSearcher.search(msg, model)
            messages.clear()
            time.sleep(2)
        overall_hit_ratio = edgeRagSearcher.get_total_cache_hit_ratio()
        print(f"[Overall Cache Hit Ratio] {overall_hit_ratio:.2%}")

if __name__ == "__main__":    
    inf_centroids_file ="../hotpotqa_centroids.index"
    kafka_search(inf_centroids_file)