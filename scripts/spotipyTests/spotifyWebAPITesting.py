import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import numpy as np
import requests
import os
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import random

# Public access for metadata
sp_public = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id='b7e9d6b4921e43659c467e32d013784d',
    client_secret='7c67c6b32d7443908ffcf9f7d1739d38'
))

def get_metadata_vector(track):
    artist_id = track['artists'][0]['id']
    artist = sp_public.artist(artist_id)
    album_release_year = int(track['album']['release_date'][:4]) if track['album']['release_date'] else 2000
    genres = set(artist['genres'])
    return np.array([
        track['popularity'],
        track['duration_ms'] / 60000,  # minutes
        int(track['explicit']),
        artist['popularity'],
        artist['followers']['total'] / 1e6,  # scaled followers
        album_release_year
    ]), genres

# Get reference track info
ref_track_name = input("Enter a reference song name: ")
results = sp_public.search(q=ref_track_name, type='track', limit=1)
if not results['tracks']['items']:
    print("No reference track found.")
    exit()

ref_track = results['tracks']['items'][0]
ref_vector, ref_genres = get_metadata_vector(ref_track)
print(f"Reference track: {ref_track['name']} by {ref_track['artists'][0]['name']}")

# Expand candidate pool across years
years = ['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022','2023']
candidates = []
for year in years:
    year_tracks = sp_public.search(q=f'year:{year}', type='track', limit=20)['tracks']['items']
    candidates.extend(year_tracks)
time.sleep(0.5)  # pause to respect API rate limits

# Collect candidate vectors and genres
candidate_data = []
for track in candidates:
    try:
        vec, genres = get_metadata_vector(track)
        genre_overlap = len(ref_genres & genres)
        vec = np.append(vec, genre_overlap)
        candidate_data.append((track, vec))
        time.sleep(0.1)
    except Exception as e:
        print(f"Skipping track {track['name']} due to error: {e}")

# Add genre overlap to reference vector
ref_vector = np.append(ref_vector, 0)

# Combine all vectors for normalization
all_vectors = np.vstack([ref_vector] + [vec for _, vec in candidate_data])
scaler = MinMaxScaler()
all_vectors_scaled = scaler.fit_transform(all_vectors)

ref_vector_scaled = all_vectors_scaled[0]
candidate_vectors_scaled = all_vectors_scaled[1:]

# Feature weights
weights = np.array([1.0, 0.5, 0.1, 1.0, 0.3, 0.2, 2.0])

# Compute similarity scores
candidate_scores = []
for i, (track, vec_scaled) in enumerate(zip([c[0] for c in candidate_data], candidate_vectors_scaled)):
    weighted_diff = (ref_vector_scaled - vec_scaled) * weights
    euclidean_score = np.linalg.norm(weighted_diff)
    cosine_score = cosine_similarity(ref_vector_scaled.reshape(1, -1), vec_scaled.reshape(1, -1))[0][0]
    combined_score = euclidean_score - cosine_score  # combine: lower is better
    candidate_scores.append((track, euclidean_score, cosine_score, combined_score))

# Sort by combined similarity score
candidate_scores.sort(key=lambda x: x[3])

# Remove duplicates by track name + artist
seen = set()
unique_scores = []
for track, euc, cos, comb in candidate_scores:
    key = (track['name'].lower(), track['artists'][0]['name'].lower())
    if key not in seen:
        unique_scores.append((track, euc, cos, comb))
        seen.add(key)

# Optionally shuffle top candidates for diversity
top_results = unique_scores[:10]
random.shuffle(top_results)

# Print top 5 diverse recommendations and download previews
output_folder = 'data/previews'
os.makedirs(output_folder, exist_ok=True)

for i, (track, euclidean_score, cosine_score, combined_score) in enumerate(top_results[:5]):
    name = track['name']
    artist = track['artists'][0]['name']
    preview_url = track['preview_url']

    print(f"\n{i+1}. {name} by {artist}")
    print(f"Euclidean similarity (lower = better): {euclidean_score:.4f}")
    print(f"Cosine similarity (higher = better): {cosine_score:.4f}")
    print(f"Combined score (lower = better): {combined_score:.4f}")
    print(f"Preview URL: {preview_url}")

    if preview_url:
        preview_path = os.path.join(output_folder, f"{name.replace(' ', '_')}_{artist.replace(' ', '_')}.mp3")
        try:
            r = requests.get(preview_url)
            with open(preview_path, 'wb') as f:
                f.write(r.content)
            print(f"Downloaded preview to {preview_path}")
        except Exception as e:
            print(f"Failed to download preview: {e}")