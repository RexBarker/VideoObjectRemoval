# VideoObjectRemoval
### Video inpainting with automatic object detection
---

### Quick Links ###

  - [Setup](./SETUP.md)

  - [Background](./DESCRIPTION.md)

  - [References](./REFERENCES.md)

### Overview
In this project, a prototype video editing system based on “inpainting” is demonstrated. Inpainting is an image editing method for replacement of masked regions of an image with a suitable background. The resulting video is thus free from the selected objects and more readily adaptable for further simulation. The system utilizes a three-step approach to simulation: (1) detection, (2) mask grouping, and (3) inpainting. The detection step involves the identification of objects within the video based upon a given object class definition, and the production of pixel level masks. Next, the object masks are grouped and tracked through the frame sequence to determine persistence and allow correction of classified results. Finally, the grouped masks are used to target specific objects instances in the video for inpainting removal.

The end result of this project is a video editing platform in the context of locomotive route simulation. The final video output demonstrates the system’s ability to automatically remove moving pedestrians in a video sequence, which commonly occur in most street tram simulations. This work also addresses the limitations of the system, in particular, the inability to remove quasi-stationary objects. The overall outcome of the project is a video editing system with automation capabilities rivaling commercial inpainting software.

### Project Video

<a href="http://www.youtube.com/watch?feature=player_embedded&v=sABfRj50FK4
" target="_blank"><img src="http://img.youtube.com/vi/sABfRj50FK4/0.jpg" 
alt="Video Object Removal Project" width="480" height="360" border="10" /></a>

### Project Results

| Result | Description  |
| ------ |:------------ |
| <img src="assets/CCperson_org_seq_inp_vert.gif" style="width:50%"/>  | **Multiple object removal** <br> - pedestrians are removed using bounding-box shaped mask  <br> - elliptical dialation mask of 21 pixels used |

---

| Result | Description  |
| ------ |:------------ |
| <img src="assets/CCCar_org_seq_inp_vert.gif" style="width:50%"/>  | **Single object removal** <br> - A single vehichle is removed using a conforming mask <br> - elliptical dialation mask of 21 pixels used |    


