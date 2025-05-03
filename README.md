# MusicAI

## Design Overview

### Purpose
This project is meant as an expansion of current Spotify AI recommendation capabilities. As of now, Spotify does not offer an advanced, configurable song recommendation system powered by modern machine learning principles. Instead, Spotify relies primarily on collaborative filtering, which works by cross-referencing various users’ listening histories to suggest tracks[1]. While this approach is successful at discovering popular or behaviorally similar content, it falls short in several important respects, including adapting to particular niches, handling limited listening histories, and providing reasoning behind recommendations.

By integrating a nearest neighbor recommendation system, this project aims at avoiding the limitations in Spotify's current algorithm. Furthermore, it contains greater functionality for music analysis, by training models to understand audio features (like tempo, key, mood, timbre, genre) through CNN-LSTM trained on WAV file mel-spectrograms. This software features various capabilities from recommendation programs to audio anaylsis.

### Original Design Concepts
Concept 1: AI Spotify Recommendation Software

The intial idea for this project was a fully Spotify based software for song recommendation, playlist creation, etc. that could be prompted by track inputs, keywords, and more. The project does still contain a lot of this functionality, but certain capabilities had to be reduced, altered, or even expanded upon due to Spotify's recent alterations to its Web API. In the past, insightful features like tempo, timbre, and loudness as well as 30 second audio snippets could be extracted for analysis, but due to licensing agreements these features were removed. While the intial concept for a multi-input WAV mel-spectrogram CNN-LSTM and track feature random forest based model couldn't be implemented due to Spotify's removal of those two key aspects of their API, new concepts for recommendation and analysis models of music were developed.

Final Design: Multifacited Music Software

Since Spotify no longer provides appropriate data for this project's purposes, the goal needed to be shifted. Instead of making a Spotify based software application, this project is focused on a wholistic development of machine learning as a means music recommendation and analysis. Ultimately there are two main apps inside this software

The first is a Spotify API song recommender. It works through a nearest neighbor approach, comparing euclidean and cosine distances between important metadata like popularity and genre between tracks. The app allows for adjustment of a "Similarity Score"; this allows the user to customize just how similar they want their recommendations to be to their inputted track. This was achieved by allowing comparison between less popular songs, as well fewer restrictions on comparison cross genres for lower similarity scores as compared to higher ones.

The second app is a WAV audio file analysis application. Through extensive CNN-LSTM training, this project generated models to provide insight on genre and feature information of an audio file. It uses an ensemble approach to combine a categorical genre classifier and linear metadata predicter. Ultimately, the user's able to input their own audio files to get insightful information they may have not been able to identify themselves.

![GenreClassifierAccuracy](https://github.com/user-attachments/assets/b00d3cd9-ef9d-42a7-81ef-4fd1f83dbe28)
![GenreClassifierLoss](https://github.com/user-attachments/assets/2f921dcd-3e57-46fd-b8c8-51743e35496e)


### Knowledge Required for Project
A prior knowledge of machine learning structures like CNN and nearest neighbors was necessary in the development of this project. It also required a baseline understanding of how machines interpret and learn from data. Computer's are reliant on the information provided to them and cannot infer about new, previously never seen, information. In other words, the data used to train the models must also be available at the time of running the software for the user. Considering the legal issue of accessing audio files in mass, it was difficult for the models to be trained on relevant, popular music. For this reason, the GTZAN genre classification dataset was used for the training of the CNN-LSTMs[2]. This dataset was essential in providing the models enough information to properly learn to interpret audio WAV files. Furthermore, thank you to EsratMaria's MusicGenreRecognition[3] github for some inspiration for the model structure.

Additionally, the development of the UI was done through the PyQt6 library. Having never developed UI before, the process was somewhat tedious but overall a fantastic learning experience. For access to both the applications, please click the "apps" button in the top left, then select an app of your choosing.

---

## Preliminary Design Verification

The first step in development was the compilation and analysis of data to gain a deeper understanding of the possibilities for machine learning in music analysis. Research into datasets such as the Million Song Dataset [4], GTZAN, and the Spotify Web API [5] led to the decision to shift focus away from audio file analysis for song recommendation and toward feature-based analysis. The limited availability of audio files across a wide variety of songs makes it difficult to use audio WAV files as inputs to a trained model unless explicitly provided by the user. However, metadata is readily available for most, if not all, songs through APIs, making a strictly metadata-based recommendation system the ideal choice. Still, the GTZAN dataset provides access to WAV files, allowing it to serve as a useful resource for audio analysis. After this analysis, it was decided the software would feature two applications using different data sources.

### Test Plan and Procedures
The testing for this project was done and documented in the scripts folder and development.txt file. In the scripts folder, the API was tested first through some basic programs to get a grasp of how to extract information about songs and artists. A feature extraction function was developed as a stepping stone towards the rest of the functionality. Once metadata was easily extractable, it was possible to compare aspects like popularity, artist genres, and follower count across songs. By choosing numerical features, the "distances" could be calculated between features of different songs to get an overall simialrity rating. After understanding how to compare tracks, it was necessary to choose good candidates to compare with. Aspects like genre, release year, and top tracks were used to select candidate tracks to compare to the user input track.

Next it was essential to test the GTZAN dataset with various models to find an ideal manner of analyzing WAV files for good genre classification and feature extraction. Initially, the aim for genre classification was an ensemble method between a CNN-LSTM to analyse WAV files and a Random Forest to analyse metadata. However, it was determined it'd be too much to expect users to input metadata information for their songs. Instead, the CNN-LSTM model trained for genre classification is used for genre classification independently, and a second CNN-LSTM model was trained for feature extraction. This way a user is able to input an audio WAV file and get a genre classification as well as information of features like harmony, tempo, pitch, loudness, and more. After importing data from the GTZAN dataset, a model structure inspired by [3] was used then analysis of the models were performed.

Finally, some UI testing using PyQt6 was done. Initially, the UI was going to be constructed of buttons across the screen, but it was much easier and cleaner to use the menu function to navigate to different areas of the software.

---

## Design Implementation

### Software Interface Overview
The interface allows the user to easily interact with all featured components of this software. When opening the UI, the user is greeted to the MusicAI software. They can then select from a menu bar at the top containing "File", "Apps", and "Help" menus. File is used just to exit the software and Help gives information on the state of the project. In Apps the user can select between a song recommender and song analyzer. These each bring the user to a new window. In the song recommender the user can input a song and similarity score to get five recommended songs and links to their Spotify. The song analyzer allows the user to upload a WAV file where the models will give a genre classification and information about features.

### Relevant Subcomponents
The main aspects of this software are organized in the src folder. This includes UI, song recommendation programs, and song analysis programs. The UI is organized into applicationUI.py and applicationWidgets.py files. The applicationUI.py file is used as the overcompassing final file to run. It gives the user access to all intended features.

The spotipyFunctions folder contains the song recommendation program, and GTZANModelTrainers contains the final group of programs that train the models. These files aren't explicitly used while running the software, but they were used to create the models stored in models/modelFinals. These models are accessed when the user inputs a WAV file during genre classification and feature analysis.

### Notable Design Practices and Challenges
This project is structured with the intent of professionalism. It separates key components of the software between development stages and final product. Notably, the data used for training, like GTZAN, are not included in the Github page. This data is present locally, but with its size it was impractical to push to Github.

---

## Design Testing

### Final Test Plan and Procedures
After developing the features, i.e. song recommendation and genre classification/feature analysis, the UI was developed for ease of use for a user of the software. After playing with the song recommendation section, it was clear more diversity in the program's recommendations would be ideal. This is when the similarity score was implemented. Furthermore, links where added to the output so the user can easily access their recommendation via Spotify.

### Testing Results
Testing on trained models were done during post-training analysis and turned into graphs. For example, one of the accuracy vs epochs training figures created during testing of model strcutures is provided below.
![CNNWAVAcc](https://github.com/user-attachments/assets/126442d3-5e95-4611-b8f4-bef4b13bc297)
Testing for the CNN-LSTM models was done immediately after training, but since the song recommendation program isn't a trained program it needed to be tested manually. This was done by developer testing and adjustments of weights and features considered. For example, candidate tracks are picked based off year and genre. After testing, it was clear results were much better when considering only tracks within a 5 year release period of the user input track.

### Debugging and Challenges
When dealing with large datasets like GTZAN or APIs there is often issues importing and formatting data. Initially, images of mel-spectrograms from the GTZAN dataset were being input to the CNN-LSTM, but it was producing terrible accuracy results. After realising the librosa library could be used for raw data (frequency) extraction of WAV file then mel-spectrogram conversion the model started having much better results. Other important steps like batching were taken to speed up training as well. Despite speeding up the process some models still managed to take quite a while, like the feature analysis model.
![image](https://github.com/user-attachments/assets/12356790-b2a5-4e9d-8a6e-0815e49ad4da)


### Unsuccessful Attempts
A great amount of effort was put into implementing GPU instruction execution to speed up training; however after many unsuccessful attempts to use CUDA with the cuDNN library[6] GPU use was deemed unnecessary. CUDA helps the GPU understand python instructions, but likely due to version mismatches CUDA was unable to identify a GPU.

---

## Summary, Conclusions, and Future Work

### Summary
Overall, this project required immense research and went through many feature adjustments. Certain features like WAV file analysis for song recommendation couldn't be implemented due to a lack of appropriate data, but new features like WAV file metadata analysis was implemented in its place. This project encapsulates the capabilities, and shortcomings, of some interesting machine learning and algorithmic models. With more data much more could be accomplished, and new features could always be added in imaginative ways.

### Future Work
With more time, features could be continually added from playlist creation based on prompts to mel-spectrogram generation from metadata inputs. Machine learning could be used to improve many of these features. Greater amounts of data available opens up roads to new ways for machine learning development, but in instances where that data isn't availabe non-machine learning based algorithms can be implemented like in the nearest neighbor approach used for song recommendations in this project.

---

**References**
[1] https://sander.ai/2014/08/05/spotify-cnns.html
[2] https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
[3] https://github.com/EsratMaria/MusicGenreRecogniton/blob/master/GenreClassificationWithCNN-LSTM.ipynb
[4] http://millionsongdataset.com/
[5] https://developer.spotify.com/documentation/web-api
[6] https://developer.nvidia.com/cuda-toolkit
