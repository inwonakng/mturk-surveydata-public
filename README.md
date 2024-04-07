## Summary

This repository contains code written by Farhad Mohsin and Inwon Kang for the moral dilemma project

The directories are as follows:
### 1_airplane_scenario/2_plant_scenario:
- html files used for survey rendering each scenario to the mturk participant.
- Contains multiple versions, `ultimate.html` being the final version



All the python files should be ran in python3
### Scripts

##### classes.json
contains seed data to combine and create random stories from.

##### various storygen.py
uses data in classes.json to create random stories of different types (purely random, focused on one attribute, etc) to randomly generate features for the survey questions

##### LP_.. files
Deals with constructing and testing the Lexicographic Preference model

### Data
All the data is stored under the data folder. 
Files containing sensitive information such as worker id have been removed, and can be accessed in a private folder.
They are divided by each round of batch that we sent them out on, and the cumulative data is in the one outside of the rounds folders.

**3_option.csv**
- Contains the survey results where participants are asked to rank from 3 possible candidates to give the life jacket to.
- `Question ID` column denotes each unique questions.
- `[AGE, GENDER, PURPOSE OF TRIP, CAREER, HEALTH, SURVIVAL]` columns denote the features of each candidates that are being picked. 
- `Survival` denotes the survival chances of each participants with/without the life jacket.
- `FOCUS` denotes the aspect the survey focused on. It can be blank (random), survival, purpose of trip and career.
- `[Age Group, Gender, Education]` columns denote the demographic information of the participants.
- `Written Response` contains an optional written response from the participants that explain their choices.
- `[A score, B score, C score]` denote the score (out of 10) assigned to each candidate by the participants.
- `Ranking` contains the ranking parsed based on the scores.


### notes.md
Previous notes on the analysis on the data are stored here
