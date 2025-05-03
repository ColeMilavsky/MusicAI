![GenreClassifierAccuracy](https://github.com/user-attachments/assets/687e3927-b7fc-4df4-a8b2-9825910b1d17)# MusicAI

## Design Overview

### Purpose
This project is meant as an expansion of current Spotify AI recommendation capabilities. As of now, Spotify does not offer an advanced, configurable song recommendation system powered by modern machine learning principles. Instead, Spotify relies primarily on collaborative filtering, which works by cross-referencing various users’ listening histories to suggest tracks[1]. While this approach is successful at discovering popular or behaviorally similar content, it falls short in several important respects, including adapting to particular niches, handling limited listening histories, and providing reasoning behind recommendations.

By integrating a nearest neighbor recommendation system, this project aims at avoiding the limitations in Spotify's current algorithm. Furthermore, it contains greater functionality for music analysis, by training models to understand audio features (like tempo, key, mood, timbre, genre) through CNN-LTSM trained on WAV file mel-spectrograms. This software features various capabilities from recommendation programs to audio anaylsis.

### Original Design Concepts
Concept 1: AI Spotify Recommendation Software

The intial idea for this project was a fully Spotify based software for song recommendation, playlist creation, etc. that could be prompted by track inputs, keywords, and more. The project does still contain a lot of this functionality, but certain capabilities had to be reduced, altered, or even expanded upon due to Spotify's recent alterations to its Web API. In the past, insightful features like tempo, timbre, and loudness as well as 30 second audio snippets could be extracted for analysis, but due to licensing agreements these features were removed. While the intial concept for a multi-input WAV mel-spectrogram CNN-LTSM and track feature random forest based model couldn't be implemented due to Spotify's removal of those two key aspects of their API, new concepts for recommendation and analysis models of music were developed.

Final Design: Multifacited Music Software

Since Spotify no longer provides appropriate data for this project's purposes, the goal needed to be shifted. Instead of making a Spotify based software application, this project is focused on a wholistic development of machine learning as a means music recommendation and analysis. Ultimately there are two main apps inside this software

The first is a Spotify API song recommender. It works through a nearest neighbor approach, comparing euclidean and cosine distances between important metadata like popularity and genre between tracks. The app allows for adjustment of a "Similarity Score"; this allows the user to customize just how similar they want their recommendations to be to their inputted track. This was achieved by allowing comparison between less popular songs, as well fewer restrictions on comparison cross genres for lower similarity scores as compared to higher ones.

The second app is a WAV audio file analysis application. Through extensive CNN-LTSM training, this project generated models to provide insight on genre and feature information of an audio file. It uses an ensemble approach to combine a categorical genre classifier and linear metadata predicter. Ultimately, the user's able to input their own audio files to get insightful information they may have not been able to identify themselves.

![GenreClassifierAccuracy](https://github.com/user-attachments/assets/b00d3cd9-ef9d-42a7-81ef-4fd1f83dbe28)
![GenreClassifierLoss](https://github.com/user-attachments/assets/2f921dcd-3e57-46fd-b8c8-51743e35496e)


### Knowledge Required for Project
A prior knowledge of machine learning structures like CNN and nearest neighbors was necessary in the development of this project. It also required a baseline understanding of how machines interpret and learn from data. Computer's are reliant on the information provided to them and cannot infer about new, previously never seen, information. In other words, the data used to train the models must also be available at the time of running the software for the user. Considering the legal issue of accessing audio files in mass, it was difficult for the models to be trained on relevant, popular music. For this reason, the GTZAN genre classification dataset was used for the training of the CNN-LTSMs[2]. This dataset was essential in providing the models enough information to properly learn to interpret audio WAV files. Furthermore, thank you to EsratMaria's MusicGenreRecognition[3] github for some inspiration for the model structure.

Additionally, the development of the UI was done through the PyQt6 library. Having never developed UI before, the process was somewhat tedious but overall a fantastic learning experience. For access to both the applications, please click the "apps" button in the top left, then select an app of your choosing.

---

## Preliminary Design Verification

### Verification Summary
The first step in development was a compilation and analysis of data to get a greater understanding of the possibilities for machine learning in music analysis.

### Test Plan and Procedures
- Describe how you tested subsystems or critical components
- Outline test cases and how you measured success

### Outcomes
- Successful tests and results
- Unsuccessful tests and lessons learned

---

## Design Implementation

### Overall System Overview
- Provide a high-level explanation of your final system
- Describe major hardware/software modules and how they interact

### Relevant Subcomponents
- **Software**
  - Module 1 → description
  - Module 2 → description

### Notable Design Practices and Challenges
- Describe important design practices used (e.g., modularity, optimization)
- Summarize challenges faced and how you overcame them

---

## Design Testing

### Final Test Plan and Procedures
- Describe the test plan for the final prototype
- Explain how you measured success for each subsystem and the overall system

### Testing Results
- List key outcomes (pass/fail, benchmarks, etc.)
- Provide context for what worked and what didn’t

### Debugging and Challenges
- Document issues encountered and how you resolved them
- Include examples of debugging methods you used

### Unsuccessful Attempts
- Describe failures and explain how you addressed or mitigated them

### Known Issues or Limitations
- If applicable, explain why the system is not fully functional or has limitations

**Photos / Screenshots**
- ![Final Prototype](path/to/final_photo.png)
- ![Test Result](path/to/test_result.png)

**Video Demonstrations**
- [YouTube or Canvas video link](https://your-video-link)

---

## Summary, Conclusions, and Future Work

### Summary
- Briefly recap the project’s goals, methods, and outcomes

### Conclusions
- Share your main takeaways and performance assessment

### Future Work
- Describe improvements, changes, or next steps you would take if you iterated on the design again

---

**References**
[1] https://sander.ai/2014/08/05/spotify-cnns.html
[2] https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
[3] https://github.com/EsratMaria/MusicGenreRecogniton/blob/master/GenreClassificationWithCNN-LSTM.ipynb
