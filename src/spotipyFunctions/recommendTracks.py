import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
import sys
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import random

# Function to get important metadata for comparisons
def get_track_metadata(track, artist_data):
    artist = artist_data[track['artists'][0]['id']]
    if track['album']['release_date']:
        album_release_year = int(track['album']['release_date'][:4])
    else:
        album_release_year = 2000
    genres = set(artist['genres'])
    return np.array([
        track['popularity'],
        track['duration_ms'] / 60000,
        int(track['explicit']),
        artist['popularity'],
        artist['followers']['total'] / 1e6,
        album_release_year
    ]), genres

def runAPIRec(ref_track_name, sim_score):
    # Run client identification
    sp_public = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id='b7e9d6b4921e43659c467e32d013784d',
        client_secret='7c67c6b32d7443908ffcf9f7d1739d38'
    ))

    results = sp_public.search(q=ref_track_name, type='track', limit=1)
    if not results['tracks']['items']:
        exit()

    # Get metadata from user chosen track
    ref_track = results['tracks']['items'][0]
    ref_vector, ref_genres = get_track_metadata(ref_track, artist_data={ref_track['artists'][0]['id']: sp_public.artist(ref_track['artists'][0]['id'])})
    print(f"Recommendations for reference track: {ref_track['name']} by {ref_track['artists'][0]['name']}\n")

    # Create a 10 year range of potential tracks to compare
    release_year_str = ref_track['album']['release_date'][:4]
    release_year = int(release_year_str)
    years = [str(release_year + offset) for offset in range(-5, 6)]

    # List of all genres in Spotify's API, used for diversification of candidates
    all_genres = [
        'acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime', 'black-metal',
        'bluegrass', 'blues', 'bossanova', 'brazil', 'breakbeat', 'british', 'cantopop',
        'chicago-house', 'children', 'chill', 'classical', 'club', 'comedy', 'country', 'dance',
        'dancehall', 'death-metal', 'deep-house', 'detroit-techno', 'disco', 'disney',
        'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro',
        'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'grunge',
        'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop',
        'holidays', 'honky-tonk', 'house', 'idm', 'indian', 'indie', 'indie-pop', 'industrial',
        'iranian', 'j-dance', 'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin',
        'latino', 'malay', 'mandopop', 'metal', 'metal-misc', 'metalcore', 'minimal-techno',
        'movies', 'mpb', 'new-age', 'new-release', 'opera', 'pagode', 'party', 'philippine-opm',
        'piano', 'pop', 'pop-film', 'post-dubstep', 'power-pop', 'progressive-house', 'psych-rock',
        'punk', 'punk-rock', 'r-n-b', 'rain', 'reggae', 'reggaeton', 'road-trip', 'rock',
        'rock-n-roll', 'rockabilly', 'romance', 'sad', 'salsa', 'samba', 'sertanejo',
        'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'soundtracks',
        'spanish', 'study', 'summer', 'swedish', 'synth-pop', 'tango', 'techno', 'trance',
        'trip-hop', 'turkish', 'work-out', 'world-music'
    ]

    possible_sim_scores = [5, 4, 3, 2, 1, 0]

    # Find index based on sim_score once
    if sim_score in possible_sim_scores:
        index = possible_sim_scores.index(sim_score)
    else:
        print("Invalid similarity score provided!")
        sys.exit()

    if not ref_genres:
        ref_genres = {'pop'}

    # Randomly select genres, more random genres selected with a lower similarity score
    sampled_genres = random.sample(all_genres, index)
    combined_genres = list(set(sampled_genres) | ref_genres)

    # Compile vector of "candidates", or other tracks, to compare metadata
    candidates = []
    for year in years:
        for _ in range(5):
            # Take less and less popular candidates as similarity score decreases
            if sim_score == 5:
                random_offset = random.randint(0, 50)
            else:
                random_offset = random.randint(0, index * 100)

            random_genre = random.choice(combined_genres)

            query = f'year:{year} genre:"{random_genre}"'
            year_tracks = sp_public.search(q=query, type='track', limit=5, offset=random_offset)['tracks']['items']
            if not year_tracks:
                # If not enough songs in a certain genre just use year as query
                fallback_query = f'year:{year}'
                tracks = sp_public.search(q=fallback_query, type='track', limit=4, offset=random_offset)['tracks']['items']
            candidates.extend(year_tracks)


    # Collect unique artist IDs and generate batches to compare user chosen track with
    artist_ids = list(set([track['artists'][0]['id'] for track in candidates]))
    artist_data = {}
    for i in range(0, len(artist_ids), 50):
        batch_ids = artist_ids[i:i + 50]
        artists = sp_public.artists(batch_ids)['artists']
        artist_data.update({artist['id']: artist for artist in artists})

    # Collect candidate metadata
    candidate_data = []
    for track in candidates:
        vec, genres = get_track_metadata(track, artist_data)
        genre_overlap = len(ref_genres & genres)
        vec = np.append(vec, genre_overlap)
        candidate_data.append((track, vec))

    # Prepare reference and candidate tracks for comparison
    ref_vector = np.append(ref_vector, 0)
    all_vectors = np.vstack([ref_vector] + [vec for _, vec in candidate_data])
    scaler = MinMaxScaler()
    all_scaled = scaler.fit_transform(all_vectors)

    ref_scaled = all_scaled[0]
    candidate_scaled = all_scaled[1:]

    # Weights correspond to the following metadata
    # Index 0: popularity
    # Index 1: duration
    # Index 2: explicit
    # Index 3: artist popularity
    # Index 4: artist followers
    # Index 5: album release year
    # Index 6: genre overlap (number of genres each artist shares)
    weights = np.array([.9, 0.2, 0.3, 1.0, 0.6, 0.3, 2.0])

    # Compare reference and candidates and compile a euclidean, cosine, and combined similarity score
    candidate_scores = []
    for (track, vec_scaled) in zip([c[0] for c in candidate_data], candidate_scaled):
        weighted_diff = (ref_scaled - vec_scaled) * weights

        # Calculate distance scores
        euclidean_score = np.linalg.norm(weighted_diff)
        cosine_score = cosine_similarity(ref_scaled.reshape(1, -1), vec_scaled.reshape(1, -1))[0][0]
        combined_score = euclidean_score - cosine_score

        candidate_scores.append((track, euclidean_score, cosine_score, combined_score))

    # Sort candidates by combined euclidean and cosine scores
    candidate_scores.sort(key=lambda x: x[3])

    # # Remove duplicates
    # seen = set()
    # unique_scores = []
    # for track, euc, cos, comb in candidate_scores:
    #     key = (track['name'].lower(), track['artists'][0]['name'].lower())
    #     if key not in seen:
    #         unique_scores.append((track, euc, cos, comb))
    #         seen.add(key)

    # Only recommend one song from a particular artist
    seen_artists = set()
    unique_results = []
    for track, euc, cos, comb in candidate_scores:
        artist_name = track['artists'][0]['name']
        if artist_name.lower() not in seen_artists:
            unique_results.append((track, euc, cos, comb))
            seen_artists.add(artist_name.lower())

    # Shuffle top candidates for more variety across runs
    top_results = unique_results[:10]
    random.shuffle(top_results)

    # Path to previews for audio URLs
    output_folder = 'data/previews'

    # Show the top 5 most similar tracks with their scores listed
    for i, (track, euc, cos, comb) in enumerate(top_results[:5]):
        name = track['name']
        artist = track['artists'][0]['name']
        spotify_link = track['external_urls']['spotify']

        print(f"<b>{i+1}. {name} by {artist}</b><br>")
        print(f"<a href='{spotify_link}'>Listen on Spotify</a><br><br>")
        #print(f"\n{i+1}. {name} by {artist}")
        #print(f"Spotify Link: {spotify_link}")