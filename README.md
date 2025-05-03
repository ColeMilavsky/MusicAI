![GenreClassifierAccuracy](https://github.com/user-attachments/assets/687e3927-b7fc-4df4-a8b2-9825910b1d17)# MusicAI

## Design Overview

### Purpose
This project is meant as an expansion of current Spotify AI recommendation capabilities. As of now, Spotify does not offer an advanced, configurable song recommendation system powered by modern machine learning principles. Instead, Spotify relies primarily on collaborative filtering, which works by cross-referencing various users’ listening histories to suggest tracks[1]. While this approach is successful at discovering popular or behaviorally similar content, it falls short in several important respects, including adapting to particular niches, handling limited listening histories, and providing reasoning behind recommendations.

By integrating a nearest neighbor recommendation system, this project aims at avoiding the limitations in Spotify's current algorithm. Furthermore, it contains greater functionality for music analysis, by training models to understand audio features (like tempo, key, mood, timbre, genre) through random-forest metadata and CNN-LTSM WAV file mel-spectrograms. This software features various capabilities from recommendation programs to audio anaylsis.

### Original Design Concepts
Concept 1: AI Spotify Recommendation Software

The intial idea for this project was a fully Spotify based software for song recommendation, playlist creation, etc. that could be prompted by track inputs, keywords, and more. The project does still contain a lot of this functionality, but certain capabilities had to be reduced, altered, or even expanded upon due to Spotify's recent alterations to its Web API. In the past, insightful features like tempo, timbre, and loudness as well as 30 second audio snippets could be extracted for analysis, but due to licensing agreements these features were removed. While the intial concept for a multi-input WAV mel-spectrogram CNN-LTSM and track feature random forest based model couldn't be implemented due to Spotify's removal of those two key aspects of their API, new concepts for recommendation and analysis models of music were developed.

Final Design: Multifacited Music Software

Since Spotify no longer provides appropriate data for this project's purposes, the goal needed to be shifted. Instead of making a Spotify based software application, this project is focused on a wholistic development of machine learning as a means music recommendation and analysis. Ultimately there are two main apps inside this software

The first is a Spotify API song recommender. It works through a nearest neighbor approach, comparing euclidean and cosine distances between important metadata like popularity and genre between tracks. The app allows for adjustment of a "Similarity Score"; this allows the user to customize just how similar they want their recommendations to be to their inputted track. This was achieved by allowing comparison between less popular songs, as well fewer restrictions on comparison cross genres for lower similarity scores as compared to higher ones.

The second app is a WAV audio file analysis application. Through extensive CNN-LTSM training, this project generated models to provide insight on genre and feature information of an audio file. 

![GenreClassifierAccuracy](https://github.com/user-attachments/assets/b00d3cd9-ef9d-42a7-81ef-4fd1f83dbe28)


### Building on Previous Work
Explain how your project expands on prior work:
- Reference past projects, papers, or tools you built upon
- Justify why this is appropriate as a 6-week project

**References**
[1] https://sander.ai/2014/08/05/spotify-cnns.html

**Images/Schematics**
- ![System Diagram](path/to/diagram.png)
- ![Schematic](path/to/schematic.png)

---

## Preliminary Design Verification

### Verification Summary
Describe how you tested early versions or prototypes:
- Breadboards, simulation tools, test programs, etc.
- How you evaluated feasibility and made early design decisions

### Test Plan and Procedures
- Describe how you tested subsystems or critical components
- Outline test cases and how you measured success

### Outcomes
- Successful tests and results
- Unsuccessful tests and lessons learned

**Prototype Photos**
- ![Prototype Photo](path/to/photo.png)
- ![Test Result Photo](path/to/photo.png)

---

## Design Implementation

### Overall System Overview
- Provide a high-level explanation of your final system
- Describe major hardware/software modules and how they interact

### Relevant Subcomponents
- **Hardware**
  - Component 1 → description
  - Component 2 → description
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
