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
They are divided by each round of batch that we sent them out on, and the cumultive data is in the one outside of the rounds folders.

### notes.md
Previous notes on the analysis on the data are stored here
