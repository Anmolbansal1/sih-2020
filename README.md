# SIH-20
Submission for SIH-2020 under the problem statement by Dte of IT & Cyber Security, DRDO.

## Problem Statement
Though there exists a technology for face recognition based authentication, dynamic human recognition based authentication is highly challenging. For a given entrance gate a hardware-software solution is needed to identify every unique person who enters or exit the gate, with log of all previous entry/exit time, photo/videos recorded. That means there will not be a previous history of an individual on the first entry. The system should immediately alert the security if it is a new person and the security will decide to allow/restrict that person entering inside the premises. Whereas, the system should learn from its previous history of videos/images dynamically to allow a known person. For a given size of the gate, the number of cameras with optimal resolution required is also to be worked out as part of solution. The solution should be scalable and preferably based open source.

## Solution Approach
We utilize the ensembling of facial recognition and gait recognition to identify a person.

### Technology Stack

- TensorFlow - Deep Learning Framework
- Flask - Web server framework
- HTML, CSS, JavaScript - Frontend Website

### Deep Learning Models
Models link: https://drive.google.com/drive/folders/1cyslku4RXEuAszfrs6OrmnI5_wfTZXNW?usp=sharing
