import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
import networkx as nx
from sklearn.neighbors import KernelDensity
from scipy import stats
import folium
from folium.plugins import MarkerCluster, HeatMap
import matplotlib.pyplot as plt
import json
import os
import tempfile
from functools import lru_cache
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Konfigurasi OSMnx
try:
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.timeout = 300
except AttributeError:
    try:
        ox.config(use_cache=True, log_console=False, timeout=300)
    except:
        pass

def setup_osmnx_storage():
    """
    Setup storage locations yang aman untuk OSMnx
    """
    try:
        # Buat directory yang aman untuk cache dan data
        base_dir = os.path.expanduser("~/osmnx_data")  # User home directory
        cache_dir = os.path.join(base_dir, "cache")
        data_dir = os.path.join(base_dir, "data")
        
        # Create directories jika belum ada
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Configure OSMnx dengan storage locations yang eksplisit
        try:
            ox.config(
                use_cache=True,
                cache_folder=cache_dir,
                data_folder=data_dir,
                log_console=True
            )
        except:
            pass
        
        print(f"✅ OSMnx storage configured:")
        print(f"   Cache: {cache_dir}")
        print(f"   Data: {data_dir}")
        
        return cache_dir, data_dir
        
    except Exception as e:
        print(f"❌ Error setting up storage: {e}")
        # Fallback ke temporary directory
        temp_dir = tempfile.gettempdir()
        cache_dir = os.path.join(temp_dir, "osmnx_cache")
        data_dir = os.path.join(temp_dir, "osmnx_data")
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        try:
            ox.config(
                use_cache=True,
                cache_folder=cache_dir,
                data_folder=data_dir,
                log_console=True
            )
        except:
            pass
        
        print(f"⚠️ Using temporary storage:")
        print(f"   Cache: {cache_dir}")
        print(f"   Data: {data_dir}")
        
        return cache_dir, data_dir

def load_accident_data(excel_path):
    """
    Load dan bersihkan data kecelakaan dari file Excel
    """
    try:
        print("📊 Loading data kecelakaan...")
        df = pd.read_excel(excel_path)
        
        lat_col = 'Koordinat GPS - Lintang'
        lon_col = 'Koordinat GPS - Bujur'
        
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(f"Kolom {lat_col} atau {lon_col} tidak ditemukan")
        
        # Parse tanggal jika ada
        if 'Tanggal Kejadian' in df.columns:
            df['Tanggal_Parsed'] = pd.to_datetime(df['Tanggal Kejadian'], errors='coerce')
            df['Tahun'] = df['Tanggal_Parsed'].dt.year
        
        # Bersihkan data
        df = df.dropna(subset=[lat_col, lon_col])
        df = df[(df[lat_col] != 0) & (df[lon_col] != 0)]
        df = df.drop_duplicates(subset=[lat_col, lon_col])
        
        print(f"✅ Data loaded: {len(df)} titik kecelakaan")
        return df
        
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def download_or_load_road_network_safe(area_name="Gowa, South Sulawesi, Indonesia", cache_dir=None, data_dir=None):
    """
    Download road network dengan storage yang aman
    """
    try:
        # Setup storage jika belum ada
        if cache_dir is None or data_dir is None:
            cache_dir, data_dir = setup_osmnx_storage()
        
        # Tentukan nama file cache yang aman
        safe_name = area_name.replace(",", "_").replace(" ", "_").replace("__", "_")
        cache_file = os.path.join(data_dir, f"{safe_name}_graph.graphml")
        
        print(f"🗂️ Graph cache file: {cache_file}")
        
        # Cek apakah file graf sudah ada dan bisa diakses
        if os.path.exists(cache_file) and os.access(cache_file, os.R_OK):
            print(f"🔄 Loading road network from cache: {cache_file}")
            try:
                G = ox.load_graphml(cache_file)
                print(f"✅ Successfully loaded from cache")
            except Exception as e:
                print(f"⚠️ Cache file corrupted, re-downloading: {e}")
                raise  # Will trigger download
        else:
            print(f"🌐 Downloading road network untuk {area_name}...")
            G = ox.graph_from_place(area_name, network_type='drive')
            
            # Coba save ke cache
            try:
                ox.save_graphml(G, cache_file)
                print(f"💾 Road network saved to cache: {cache_file}")
            except Exception as e:
                print(f"⚠️ Could not save to cache: {e}")
                print("Continuing without cache...")
        
        # Convert ke undirected
        G_undirected = G.to_undirected()
        
        print(f"✅ Road network ready: {len(G_undirected.nodes)} nodes, {len(G_undirected.edges)} edges")
        return G_undirected
        
    except Exception as e:
        raise Exception(f"Error loading road network with safe storage: {str(e)}")

def debug_distance_matrix(distance_matrix):
    """
    Debug distance matrix untuk memastikan ada variasi yang cukup
    """
    print(f"\n🔍 Distance Matrix Analysis:")
    print(f"   • Shape: {distance_matrix.shape}")
    print(f"   • Min distance: {np.min(distance_matrix[distance_matrix > 0]):.2f} meters")
    print(f"   • Max distance: {np.max(distance_matrix):.2f} meters") 
    print(f"   • Mean distance: {np.mean(distance_matrix[distance_matrix > 0]):.2f} meters")
    print(f"   • Std distance: {np.std(distance_matrix[distance_matrix > 0]):.2f} meters")
    
    # Check zero distances (should only be diagonal)
    zero_distances = np.sum(distance_matrix == 0)
    expected_zeros = distance_matrix.shape[0]  # Only diagonal should be zero
    if zero_distances > expected_zeros:
        print(f"⚠️  WARNING: {zero_distances - expected_zeros} unexpected zero distances found")
    
    # Check untuk duplicate points (same coordinates)
    unique_distances = len(np.unique(distance_matrix))
    total_distances = distance_matrix.shape[0] * distance_matrix.shape[1]
    print(f"   • Unique distance values: {unique_distances} of {total_distances} total")
    
    return distance_matrix

def debug_weights_and_kde(df, sample_weights, coords):
    """
    Debug function untuk memverifikasi penggunaan weights dalam KDE
    """
    print("\n" + "=" * 60)
    print("🔬 DEBUGGING WEIGHTS AND KDE USAGE")
    print("=" * 60)
    
    # 1. Analisis Sample Weights
    if sample_weights is None:
        print("❌ ERROR: sample_weights is None - KDE akan berjalan tanpa weights")
        return False
    
    print(f"📊 Sample Weights Statistics:")
    print(f"   • Shape: {sample_weights.shape}")
    print(f"   • Min: {np.min(sample_weights):.6f}")
    print(f"   • Max: {np.max(sample_weights):.6f}")
    print(f"   • Mean: {np.mean(sample_weights):.6f}")
    print(f"   • Std: {np.std(sample_weights):.6f}")
    print(f"   • Unique values: {len(np.unique(sample_weights))}")
    
    # 2. Check apakah weights uniform (tidak ada variasi)
    weight_variation = np.std(sample_weights)
    if weight_variation < 1e-10:
        print("⚠️  WARNING: Sample weights hampir uniform (std < 1e-10)")
        print("   KDE dengan weights uniform akan memberikan hasil sama dengan unweighted KDE")
        return False
    
    # 3. Visualisasi distribusi weights
    print(f"\n📈 Weight Distribution:")
    weights_sorted = np.sort(sample_weights)
    quartiles = np.percentile(weights_sorted, [25, 50, 75])
    print(f"   • Q1: {quartiles[0]:.6f}")
    print(f"   • Q2 (median): {quartiles[1]:.6f}")  
    print(f"   • Q3: {quartiles[2]:.6f}")
    
    # 4. Test KDE dengan dan tanpa weights
    print(f"\n🧪 Testing KDE with and without weights:")
    
    # KDE tanpa weights
    kde_no_weights = KernelDensity(kernel='gaussian', bandwidth=0.008)
    kde_no_weights.fit(coords)
    density_no_weights = kde_no_weights.score_samples(coords)
    
    # KDE dengan weights
    kde_with_weights = KernelDensity(kernel='gaussian', bandwidth=0.008)
    kde_with_weights.fit(coords, sample_weight=sample_weights)
    density_with_weights = kde_with_weights.score_samples(coords)
    
    # Compare results
    density_diff = np.abs(density_with_weights - density_no_weights)
    max_diff = np.max(density_diff)
    mean_diff = np.mean(density_diff)
    
    print(f"   • Max difference in log-density: {max_diff:.6f}")
    print(f"   • Mean difference in log-density: {mean_diff:.6f}")
    
    if max_diff < 1e-6:
        print("❌ PROBLEM: Weights tidak memberikan efek signifikan pada KDE")
        print("   Kemungkinan penyebab: weights terlalu uniform atau ada masalah dalam perhitungan")
        return False
    else:
        print("✅ SUCCESS: Weights memberikan efek signifikan pada KDE")
        return True

def calculate_distance_matrix_fixed(df, graph, max_workers=4):
    """
    FIXED: Perhitungan distance matrix yang benar dengan error handling
    """
    try:
        print("🔄 Calculating distance matrix with fixed implementation...")
        
        # Siapkan koordinat sebagai list of tuples
        coords = [(row['Koordinat GPS - Lintang'], row['Koordinat GPS - Bujur']) 
                 for idx, row in df.iterrows()]
        n_points = len(coords)
        
        print(f"📊 Processing {n_points} accident points...")
        
        if n_points == 0:
            raise ValueError("No valid coordinates found in data")
        
        # Pre-compute nearest nodes dengan error handling yang lebih baik
        print("📍 Finding nearest nodes for all points...")
        nearest_nodes = {}
        valid_points = 0
        
        for i, (lat, lon) in enumerate(coords):
            try:
                # Pastikan koordinat valid
                if np.isnan(lat) or np.isnan(lon):
                    nearest_nodes[i] = None
                    continue
                    
                node = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
                nearest_nodes[i] = node
                valid_points += 1
                
            except Exception as e:
                print(f"⚠️ Error finding nearest node for point {i} ({lat}, {lon}): {e}")
                nearest_nodes[i] = None
        
        print(f"✅ Found nearest nodes for {valid_points}/{n_points} points")
        
        if valid_points == 0:
            raise ValueError("No valid nearest nodes found - check if coordinates are within the area")
        
        # Initialize distance matrix
        distance_matrix = np.zeros((n_points, n_points))
        
        def calculate_batch_distances_fixed(batch_indices):
            """
            Fixed batch processing dengan robust error handling
            """
            batch_results = []
            
            for i in batch_indices:
                if nearest_nodes[i] is None:
                    # Fallback ke Euclidean untuk titik yang tidak valid
                    for j in range(n_points):
                        if i == j:
                            distance = 0
                        else:
                            lat1, lon1 = coords[i]
                            lat2, lon2 = coords[j]
                            try:
                                distance = ox.distance.euclidean_dist_vec(lat1, lon1, lat2, lon2)
                            except:
                                distance = 0
                        batch_results.append((i, j, distance))
                    continue
                
                source_node = nearest_nodes[i]
                
                try:
                    # Compute shortest path lengths dari source ke semua target
                    lengths = nx.single_source_dijkstra_path_length(
                        graph, source_node, weight='length'
                    )
                    
                    for j in range(n_points):
                        if i == j:
                            distance = 0
                        elif nearest_nodes[j] is None:
                            # Fallback ke Euclidean
                            lat1, lon1 = coords[i]
                            lat2, lon2 = coords[j]
                            try:
                                distance = ox.distance.euclidean_dist_vec(lat1, lon1, lat2, lon2)
                            except:
                                distance = 0
                        elif nearest_nodes[j] in lengths:
                            distance = lengths[nearest_nodes[j]]
                        else:
                            # Tidak ada path, gunakan Euclidean
                            lat1, lon1 = coords[i]
                            lat2, lon2 = coords[j]
                            try:
                                distance = ox.distance.euclidean_dist_vec(lat1, lon1, lat2, lon2)
                            except:
                                distance = 0
                        
                        batch_results.append((i, j, distance))
                        
                except Exception as e:
                    print(f"⚠️ Error calculating distances from point {i}: {e}")
                    # Fallback ke Euclidean untuk semua pairs
                    for j in range(n_points):
                        if i == j:
                            distance = 0
                        else:
                            lat1, lon1 = coords[i]
                            lat2, lon2 = coords[j]
                            try:
                                distance = ox.distance.euclidean_dist_vec(lat1, lon1, lat2, lon2)
                            except:
                                distance = 0
                        batch_results.append((i, j, distance))
            
            return batch_results
        
        # Bagi work ke batches
        batch_size = max(1, n_points // max_workers)
        batches = [list(range(i, min(i + batch_size, n_points))) 
                  for i in range(0, n_points, batch_size)]
        
        print(f"🔄 Processing {len(batches)} batches with {max_workers} workers...")
        
        total_pairs_processed = 0
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(calculate_batch_distances_fixed, batch): batch 
                for batch in batches
            }
            
            for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="Processing batches"):
                try:
                    batch_results = future.result()
                    
                    # Fill distance matrix
                    for i, j, distance in batch_results:
                        distance_matrix[i, j] = distance
                        total_pairs_processed += 1
                        
                except Exception as e:
                    print(f"⚠️ Batch processing error: {e}")
                    continue
        
        print(f"✅ Distance calculation complete: {total_pairs_processed} pairs processed")
        
        if total_pairs_processed == 0:
            raise ValueError("No distance pairs were successfully calculated")
        
        return distance_matrix
        
    except Exception as e:
        raise Exception(f"Error calculating distance matrix: {str(e)}")

def create_weight_matrix_with_validation(distance_matrix, coords):
    """
    Enhanced version dengan comprehensive validation
    """
    try:
        print("⚖️ Creating weight matrix with validation...")
        
        # Debug distance matrix terlebih dahulu
        distance_matrix = debug_distance_matrix(distance_matrix)
        
        n_points = len(coords)
        
        # 1. Validate distance matrix
        if distance_matrix.shape != (n_points, n_points):
            raise ValueError(f"Distance matrix shape {distance_matrix.shape} doesn't match coords {n_points}")
        
        # 2. Check for invalid distances
        if np.any(np.isnan(distance_matrix)) or np.any(np.isinf(distance_matrix)):
            print("⚠️ WARNING: Distance matrix contains NaN or Inf values")
            distance_matrix = np.nan_to_num(distance_matrix, nan=0.0, posinf=50000.0)
        
        # 3. Ensure matrix is symmetric
        if not np.allclose(distance_matrix, distance_matrix.T, rtol=1e-10):
            print("⚠️ WARNING: Distance matrix is not symmetric, making it symmetric")
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # 4. Apply exponential decay dengan parameter adjustment
        max_distance = np.max(distance_matrix)
        print(f"📏 Max distance in matrix: {max_distance:.2f} meters")
        
        if max_distance > 0:
            # Adaptive max_distance berdasarkan data
            # Gunakan 95th percentile sebagai max_distance untuk menghindari outliers
            max_distance_adaptive = np.percentile(distance_matrix[distance_matrix > 0], 95)
            print(f"📏 Using adaptive max distance (95th percentile): {max_distance_adaptive:.2f} meters")
            
            normalized_distances = np.minimum(distance_matrix / max_distance_adaptive, 1.0)
        else:
            normalized_distances = distance_matrix
        
        # 5. Apply exponential decay dengan parameter yang lebih agresif
        decay_parameter = -5  # Lebih agresif dari -3 untuk memberikan variasi lebih besar
        weight_matrix = np.exp(decay_parameter * normalized_distances)
        
        # Set diagonal to 1 (self-weight)
        np.fill_diagonal(weight_matrix, 1.0)
        
        # 6. Convert matrix ke sample weights
        sample_weights = np.sum(weight_matrix, axis=1)
        
        # 7. Normalization with better approach
        if np.sum(sample_weights) > 0:
            # Preserve relative differences, don't force uniform mean
            sample_weights = sample_weights / np.sum(sample_weights) * n_points
        else:
            print("❌ ERROR: All sample weights are zero")
            sample_weights = np.ones(n_points)
        
        # 8. Apply minimum weight dengan nilai yang lebih kecil
        min_weight = 0.01  # Lebih kecil dari 0.1 untuk preservasi variasi
        sample_weights = np.maximum(sample_weights, min_weight)
        
        # 9. Final validation
        weight_range = np.max(sample_weights) - np.min(sample_weights)
        print(f"📊 Final weight statistics:")
        print(f"   • Range: {weight_range:.6f}")
        print(f"   • Ratio (max/min): {np.max(sample_weights)/np.min(sample_weights):.2f}")
        
        if weight_range < 0.001:
            print("⚠️ WARNING: Weight range is very small, may not provide significant KDE differences")
        
        print("✅ Weight matrix created with validation")
        return sample_weights
        
    except Exception as e:
        raise Exception(f"Error creating validated weight matrix: {str(e)}")

def compare_kde_methods(df, distance_matrix=None, bandwidth=0.008, threshold=1.96):
    """
    Compare berbagai metode KDE untuk validasi hasil
    """
    print("\n" + "=" * 60)
    print("🧪 COMPREHENSIVE KDE METHOD COMPARISON")
    print("=" * 60)
    
    coords = df[['Koordinat GPS - Lintang', 'Koordinat GPS - Bujur']].values
    results = {}
    
    # Method 1: Spatial-only KDE (baseline)
    print("\n1️⃣ Spatial-only KDE (baseline):")
    kde_spatial = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde_spatial.fit(coords)
    density_spatial = np.exp(kde_spatial.score_samples(coords))
    z_scores_spatial = stats.zscore(density_spatial)
    hotspots_spatial = coords[z_scores_spatial > threshold]
    
    results['spatial'] = {
        'hotspots': len(hotspots_spatial),
        'density_mean': np.mean(density_spatial),
        'density_std': np.std(density_spatial),
        'z_score_max': np.max(z_scores_spatial)
    }
    print(f"   • Hotspots detected: {len(hotspots_spatial)}")
    print(f"   • Density range: {np.min(density_spatial):.6f} - {np.max(density_spatial):.6f}")
    
    # Method 2: Network Distance KDE (if distance matrix available)
    if distance_matrix is not None:
        print("\n2️⃣ Network Distance KDE:")
        sample_weights = create_weight_matrix_with_validation(distance_matrix, coords)
        
        # Debug weights
        weights_valid = debug_weights_and_kde(df, sample_weights, coords)
        
        if weights_valid:
            kde_network = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
            kde_network.fit(coords, sample_weight=sample_weights)
            density_network = np.exp(kde_network.score_samples(coords))
            z_scores_network = stats.zscore(density_network)
            hotspots_network = coords[z_scores_network > threshold]
            
            results['network'] = {
                'hotspots': len(hotspots_network),
                'density_mean': np.mean(density_network),
                'density_std': np.std(density_network),
                'z_score_max': np.max(z_scores_network)
            }
            print(f"   • Hotspots detected: {len(hotspots_network)}")
            print(f"   • Density range: {np.min(density_network):.6f} - {np.max(density_network):.6f}")
            
            # Compare results
            print(f"\n📊 COMPARISON RESULTS:")
            hotspot_diff = len(hotspots_network) - len(hotspots_spatial)
            density_change = (np.mean(density_network) - np.mean(density_spatial)) / np.mean(density_spatial) * 100
            
            print(f"   • Hotspot count change: {hotspot_diff:+d} ({hotspot_diff/len(hotspots_spatial)*100:+.1f}%)")
            print(f"   • Mean density change: {density_change:+.2f}%")
            
            if abs(hotspot_diff) < 1 and abs(density_change) < 1:
                print("⚠️ ISSUE: Network distances tidak memberikan perubahan signifikan")
                print("Kemungkinan masalah:")
                print("• Weights terlalu uniform")  
                print("• Distance calculation tidak akurat")
                print("• Parameter bandwidth/threshold tidak optimal")
            else:
                print("✅ SUCCESS: Network distances memberikan hasil berbeda signifikan")
        else:
            results['network'] = None
    
    # Method 3: Alternative bandwidth
    print(f"\n3️⃣ Spatial KDE dengan bandwidth alternatif:")
    alt_bandwidth = bandwidth * 2  # Test dengan bandwidth 2x
    kde_alt = KernelDensity(kernel='gaussian', bandwidth=alt_bandwidth)
    kde_alt.fit(coords)
    density_alt = np.exp(kde_alt.score_samples(coords))
    z_scores_alt = stats.zscore(density_alt)
    hotspots_alt = coords[z_scores_alt > threshold]
    
    results['alternative'] = {
        'hotspots': len(hotspots_alt),
        'bandwidth': alt_bandwidth
    }
    print(f"   • Bandwidth: {alt_bandwidth}")
    print(f"   • Hotspots detected: {len(hotspots_alt)}")
    
    return results

def perform_kde_analysis_with_optimal_params(df, sample_weights=None, bandwidth=0.008, threshold=1.96):
    """
    Perform KDE analysis dengan parameter optimal
    """
    try:
        print("🔥 Performing KDE analysis...")
        print(f"Parameters: bandwidth={bandwidth}, threshold={threshold}")
        
        coords = df[['Koordinat GPS - Lintang', 'Koordinat GPS - Bujur']].values
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        
        if sample_weights is not None:
            print("Using weighted KDE approach (network distances)")
            kde.fit(coords, sample_weight=sample_weights)
        else:
            print("Using spatial-only KDE approach")
            kde.fit(coords)
        
        coords_density = kde.score_samples(coords)
        density_exp = np.exp(coords_density)
        z_scores = stats.zscore(density_exp)
        
        hotspot_indices = np.where(z_scores > threshold)[0]
        
        hotspots = []
        for i, idx in enumerate(hotspot_indices):
            hotspots.append({
                'latitude': coords[idx][0],
                'longitude': coords[idx][1],
                'density': density_exp[idx],
                'z_score': z_scores[idx]
            })
        
        print(f"✅ KDE analysis complete: {len(hotspots)} hotspots identified")
        return hotspots, kde, coords_density, density_exp, z_scores
        
    except Exception as e:
        raise Exception(f"Error in KDE analysis: {str(e)}")

def filter_roads_for_hotspots(hotspot_coords, graph, radius_km=0.15):
    """
    Filter road segments untuk hotspot dengan radius yang ditentukan
    """
    try:
        print("🛣️ Filtering road segments near hotspots...")
        
        if len(hotspot_coords) == 0 or graph is None:
            return []
        
        from scipy.spatial import cKDTree
        hotspot_tree = cKDTree(hotspot_coords)
        
        def filter_segment(u, v, graph, radius_km):
            lat_u, lon_u = graph.nodes[u]["y"], graph.nodes[u]["x"]
            lat_v, lon_v = graph.nodes[v]["y"], graph.nodes[v]["x"]
            midpoint = [(lat_u + lat_v) / 2, (lon_u + lon_v) / 2]
            
            distances, _ = hotspot_tree.query(midpoint, k=1, distance_upper_bound=radius_km / 111.0)
            if distances < np.inf:
                return ((lat_u, lon_u), (lat_v, lon_v))
            return None
        
        road_segments = []
        for u, v, edge_data in graph.edges(data=True):
            result = filter_segment(u, v, graph, radius_km)
            if result:
                road_segments.append(result)
        
        print(f"✅ Found {len(road_segments)} road segments near hotspots")
        return road_segments
        
    except Exception as e:
        print(f"⚠️ Warning: Could not find road segments: {str(e)}")
        return []

def map_color(value, vmin, vmax):
    """Map density value ke warna Viridis"""
    if vmax == vmin:
        normalized = 0.5
    else:
        normalized = (value - vmin) / (vmax - vmin)
    normalized = max(0, min(1, normalized))
    
    cmap = plt.cm.viridis
    rgba = cmap(normalized)
    return f'#{int(rgba[0] * 255):02x}{int(rgba[1] * 255):02x}{int(rgba[2] * 255):02x}'

def get_accident_color_and_radius(tingkat_kecelakaan):
    """Get color and radius berdasarkan tingkat kecelakaan"""
    if tingkat_kecelakaan:
        level = str(tingkat_kecelakaan).strip().lower()
        
        if level in ["ringan", "rendah", "kecil"]:
            return "green", 5, "Ringan"
        elif level in ["sedang", "menengah", "moderate"]:
            return "orange", 7, "Sedang" 
        elif level in ["berat", "tinggi", "parah", "severe"]:
            return "red", 9, "Berat"
        else:
            return "green", 5, "Ringan"
    else:
        return "green", 5, "Ringan"

def create_visualization_with_accident_points_and_filters(df, hotspots, road_segments, kde_model, sample_weights=None, output_dir=tempfile.gettempdir()):
    """
    Visualisasi dengan hotspots, accident points, dan filter checkbox
    """
    try:
        print("🗺️ Creating visualization...")
        
        if len(hotspots) > 0:
            center_lat = np.mean([h['latitude'] for h in hotspots])
            center_lon = np.mean([h['longitude'] for h in hotspots])
        else:
            center_lat = df['Koordinat GPS - Lintang'].mean()
            center_lon = df['Koordinat GPS - Bujur'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        if len(hotspots) > 0:
            densities = [h['density'] for h in hotspots]
            vmin, vmax = min(densities), max(densities)
        else:
            vmin, vmax = 0, 1
        
        # LAYER 1: HOTSPOTS
        hotspot_group = folium.FeatureGroup(name="Hotspot Area")
        
        for idx, hotspot in enumerate(hotspots):
            color = map_color(hotspot['density'], vmin, vmax)
            
            popup_text = f"""
            <b>🔥 Hotspot #{idx+1}</b><br/>
            Density: {hotspot['density']:.4f}<br/>
            Z-score: {hotspot['z_score']:.2f}<br/>
            Coordinates: ({hotspot['latitude']:.6f}, {hotspot['longitude']:.6f})<br/>
            Parameters: BW=0.008, Z=1.96<br/>
            Method: {'Network Distance' if sample_weights is not None else 'Spatial KDE'}<br/>
            Road radius: 150m
            """
            
            if sample_weights is not None:
                coords = df[['Koordinat GPS - Lintang', 'Koordinat GPS - Bujur']].values
                for i, (lat, lon) in enumerate(coords):
                    if abs(lat - hotspot['latitude']) < 1e-6 and abs(lon - hotspot['longitude']) < 1e-6:
                        popup_text += f"<br/>Network Connectivity: {sample_weights[i]:.3f}"
                        break
            
            folium.CircleMarker(
                location=[hotspot['latitude'], hotspot['longitude']],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=popup_text,
                tooltip=f"Hotspot {idx+1}"
            ).add_to(hotspot_group)
        
        # LAYER 2: ACCIDENT POINTS
        accident_group = folium.FeatureGroup(name="Titik Kecelakaan")
        
        for idx, row in df.iterrows():
            tingkat_kecelakaan = row.get('Tingkat Kecelakaan', 'Ringan')
            color, radius, level_normalized = get_accident_color_and_radius(tingkat_kecelakaan)
            
            popup_text = f"""
            <b>📍 Kecelakaan #{idx+1}</b><br/>
            Tingkat: {level_normalized}<br/>
            Koordinat: ({row['Koordinat GPS - Lintang']:.6f}, {row['Koordinat GPS - Bujur']:.6f})
            """
            
            if 'Tanggal Kejadian' in row and pd.notna(row['Tanggal Kejadian']):
                popup_text += f"<br/>Tanggal: {row['Tanggal Kejadian']}"
            
            folium.CircleMarker(
                location=[row['Koordinat GPS - Lintang'], row['Koordinat GPS - Bujur']],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=popup_text,
                tooltip=f"Kecelakaan {level_normalized}"
            ).add_to(accident_group)
        
        # LAYER 3: ROAD SEGMENTS
        road_group = folium.FeatureGroup(name="Jalan Rawan")
        
        coords = df[['Koordinat GPS - Lintang', 'Koordinat GPS - Bujur']].values
        for ((lat_u, lon_u), (lat_v, lon_v)) in road_segments:
            midpoint_coords = np.array([[(lat_u + lat_v) / 2, (lon_u + lon_v) / 2]])
            midpoint_density = kde_model.score_samples(midpoint_coords)
            midpoint_density_exp = np.exp(midpoint_density[0])
            color = map_color(midpoint_density_exp, vmin, vmax)
            
            folium.PolyLine(
                locations=[(lat_u, lon_u), (lat_v, lon_v)],
                color=color,
                weight=3,
                opacity=0.7,
                tooltip=f"Road density: {midpoint_density_exp:.4f}"
            ).add_to(road_group)
        
        # ADD LAYERS TO MAP
        hotspot_group.add_to(m)
        accident_group.add_to(m)
        road_group.add_to(m)
        
        # ADD LAYER CONTROL
        folium.LayerControl(position='topright', collapsed=False).add_to(m)
        
        # ADD CUSTOM FILTER CONTROLS
        filter_html = '''
        <script>
            function toggleHotspots() {
                var checkbox = document.getElementById('toggle-hotspots');
                var map = window[Object.keys(window).find(key => key.includes('map_'))];
                
                map.eachLayer(function(layer) {
                    if (layer.options && layer.options.name === "Hotspot Area") {
                        if (checkbox.checked) {
                            if (!map.hasLayer(layer)) map.addLayer(layer);
                        } else {
                            map.removeLayer(layer);
                        }
                    }
                });
            }

            function toggleAccidents() {
                var checkbox = document.getElementById('toggle-accidents');
                var map = window[Object.keys(window).find(key => key.includes('map_'))];
                
                map.eachLayer(function(layer) {
                    if (layer.options && layer.options.name === "Titik Kecelakaan") {
                        if (checkbox.checked) {
                            if (!map.hasLayer(layer)) map.addLayer(layer);
                        } else {
                            map.removeLayer(layer);
                        }
                    }
                });
            }
            
            function toggleRoads() {
                var checkbox = document.getElementById('toggle-roads');
                var map = window[Object.keys(window).find(key => key.includes('map_'))];
                
                map.eachLayer(function(layer) {
                    if (layer.options && layer.options.name === "Jalan Rawan") {
                        if (checkbox.checked) {
                            if (!map.hasLayer(layer)) map.addLayer(layer);
                        } else {
                            map.removeLayer(layer);
                        }
                    }
                });
            }
        </script>
        '''
        
        
        return m
        
    except Exception as e:
        raise Exception(f"Error creating visualization: {str(e)}")

def create_visualization_and_output(df, hotspots, road_segments, kde_model, sample_weights=None, output_dir=tempfile.gettempdir()):
    """
    Create visualisasi dan output dengan network distance info
    """
    try:
        print(f"🗺️ Creating visualization and output...")
        
        m = create_visualization_with_accident_points_and_filters(df, hotspots, road_segments, kde_model, sample_weights, output_dir)
        
        html_path = os.path.join(output_dir, 'hotspot_map.html')
        m.save(html_path)
        
        # Create GeoJSON dengan network distance metadata
        geojson_data = {
            "type": "FeatureCollection",
            "features": [],
            "properties": {
                "analysis_method": "OSMnx Network Distance KDE" if sample_weights is not None else "Spatial KDE",
                "distance_calculation": "Dijkstra shortest path via road network" if sample_weights is not None else "Euclidean",
                "bandwidth": 0.008,
                "zscore_threshold": 1.96,
                "road_radius_km": 0.15
            }
        }
        
        if len(hotspots) > 0:
            densities = [h['density'] for h in hotspots]
            vmin, vmax = min(densities), max(densities)
        else:
            vmin, vmax = 0, 1
        
        # Add hotspots
        for i, hotspot in enumerate(hotspots):
            color = map_color(hotspot['density'], vmin, vmax)
            
            properties = {
                "category": "hotspot",
                "hotspot_id": i + 1,
                "density": float(hotspot['density']),
                "z_score": float(hotspot['z_score']),
                "color": color,
                "method": "network_distance" if sample_weights is not None else "spatial_only"
            }
            
            if sample_weights is not None:
                coords = df[['Koordinat GPS - Lintang', 'Koordinat GPS - Bujur']].values
                for j, (lat, lon) in enumerate(coords):
                    if abs(lat - hotspot['latitude']) < 1e-6 and abs(lon - hotspot['longitude']) < 1e-6:
                        properties["network_connectivity"] = float(sample_weights[j])
                        break
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [hotspot['longitude'], hotspot['latitude']]
                },
                "properties": properties
            }
            geojson_data["features"].append(feature)
        
        # Add accident points
        for idx, row in df.iterrows():
            tingkat_kecelakaan = row.get('Tingkat Kecelakaan', 'Ringan')
            color, radius, level_normalized = get_accident_color_and_radius(tingkat_kecelakaan)
            
            properties = {
                "category": "accident_point",
                "accident_id": idx + 1,
                "Tingkat Kecelakaan": level_normalized,
                "color": color,
                "radius": radius
            }
            
            if 'Tanggal Kejadian' in row and pd.notna(row['Tanggal Kejadian']):
                properties["Tanggal Kejadian"] = str(row['Tanggal Kejadian'])
            if 'Tahun' in row and pd.notna(row['Tahun']):
                properties["Tahun"] = int(row['Tahun'])
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['Koordinat GPS - Bujur'], row['Koordinat GPS - Lintang']]
                },
                "properties": properties
            }
            geojson_data["features"].append(feature)
        
        # Add road segments
        for ((lat_u, lon_u), (lat_v, lon_v)) in road_segments:
            midpoint_coords = np.array([[(lat_u + lat_v) / 2, (lon_u + lon_v) / 2]])
            midpoint_density = kde_model.score_samples(midpoint_coords)
            midpoint_density_exp = np.exp(midpoint_density[0])
            color = map_color(midpoint_density_exp, vmin, vmax)
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon_u, lat_u], [lon_v, lat_v]]
                },
                "properties": {
                    "category": "road_segment",
                    "density": float(midpoint_density_exp),
                    "color": color
                }
            }
            geojson_data["features"].append(feature)
        
        # Save GeoJSON
        geojson_path = os.path.join(output_dir, 'hotspot_results.geojson')
        with open(geojson_path, 'w') as f:
            json.dump(geojson_data, f, indent=2)
        
        print(f"✅ Visualization saved: {html_path}")
        print(f"✅ GeoJSON saved: {geojson_path}")
        
        return geojson_path, html_path
        
    except Exception as e:
        raise Exception(f"Error creating visualization: {str(e)}")

def run_analysis_with_comprehensive_validation(input_excel_path, area_name="Gowa, South Sulawesi, Indonesia", 
                                             bandwidth=0.008, threshold=1.96, max_workers=4, 
                                             output_dir=None):
    """
    MAIN ANALYSIS dengan comprehensive validation untuk memastikan weighted KDE bekerja
    """
    try:
        print("🚀 STARTING ANALYSIS WITH COMPREHENSIVE VALIDATION")
        print("=" * 60)
        
        # Setup storage yang aman
        cache_dir, data_dir = setup_osmnx_storage()
        
        # Setup output directory
        if output_dir is None:
            output_dir = data_dir  # Gunakan data directory yang sama
        
        # Load data
        df = load_accident_data(input_excel_path)
        
        # Load road network dengan storage yang aman
        graph = download_or_load_road_network_safe(area_name, cache_dir, data_dir)
        
        # Calculate distance matrix
        print("🔄 Calculating distance matrix...")
        distance_matrix = calculate_distance_matrix_fixed(df, graph, max_workers)
        
        # Comprehensive KDE comparison
        comparison_results = compare_kde_methods(df, distance_matrix, bandwidth, threshold)
        
        # Use network-based weights jika valid
        sample_weights = create_weight_matrix_with_validation(distance_matrix, df[['Koordinat GPS - Lintang', 'Koordinat GPS - Bujur']].values)
        weights_valid = debug_weights_and_kde(df, sample_weights, df[['Koordinat GPS - Lintang', 'Koordinat GPS - Bujur']].values)
        
        if not weights_valid:
            print("⚠️ WARNING: Using spatial-only KDE due to weight issues")
            sample_weights = None
        
        # Perform final KDE analysis
        hotspots, kde_model, coords_density, density_exp, z_scores = perform_kde_analysis_with_optimal_params(
            df, sample_weights, bandwidth, threshold
        )
        
        # Filter road segments
        if len(hotspots) > 0:
            hotspot_coords = np.array([[h['latitude'], h['longitude']] for h in hotspots])
            road_segments = filter_roads_for_hotspots(hotspot_coords, graph, 0.15)
        else:
            road_segments = []
        
        # Create output
        geojson_path, html_path = create_visualization_and_output(
            df, hotspots, road_segments, kde_model, sample_weights, output_dir
        )
        
        # Enhanced summary
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"🌍 Area: {area_name}")
        print(f"📊 Total Accidents: {len(df)}")
        print(f"🔥 Final Hotspots: {len(hotspots)}")
        print(f"⚙️ Method: {'Weighted Network KDE' if weights_valid else 'Spatial-Only KDE'}")
        print(f"📊 Comparison Results:")
        for method, result in comparison_results.items():
            if result:
                print(f"   • {method}: {result.get('hotspots', 'N/A')} hotspots")
        print(f"📁 Storage Locations:")
        print(f"   • Cache: {cache_dir}")
        print(f"   • Data: {data_dir}")
        print(f"   • Output: {output_dir}")
        print(f"📁 Output Files:")
        print(f"   • Map: {html_path}")
        print(f"   • Data: {geojson_path}")
        print("=" * 60)
        
        return geojson_path, html_path
        
    except Exception as e:
        raise Exception(f"Comprehensive analysis failed: {str(e)}")

def run_analysis(input_excel_path, area_name="Gowa, South Sulawesi, Indonesia", 
                bandwidth=0.008, threshold=1.96, max_workers=1, 
                output_dir=None):
    """
    Main entry point - otomatis menggunakan comprehensive validation
    """
    try:
        return run_analysis_with_comprehensive_validation(
            input_excel_path=input_excel_path,
            area_name=area_name,
            bandwidth=bandwidth,
            threshold=threshold,
            max_workers=max_workers,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"⚠️ Analysis failed: {e}")
        raise

# Test functions untuk debugging
def test_osmnx_storage():
    """
    Test OSMnx storage functionality
    """
    try:
        cache_dir, data_dir = setup_osmnx_storage()
        
        # Test write permission
        test_file = os.path.join(data_dir, "test_write.txt")
        with open(test_file, 'w') as f:
            f.write("Test write permission")
        
        # Test read permission
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Cleanup
        os.remove(test_file)
        
        print("✅ Storage test passed - read/write permissions OK")
        return True
        
    except Exception as e:
        print(f"❌ Storage test failed: {e}")
        return False
