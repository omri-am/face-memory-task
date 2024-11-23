**An Attempt to Replicate the Human Familiarity Disadvantage Effect in Neural Networks as Reported by Regine G.M. Armann, Rob Jenkins, and A. Mike Burton**

Project by Omri Amit and Mia Shlein

Modeling human memory, particularly in face recognition, remains a significant challenge in artificial intelligence and cognitive science. This study aimed to replicate the findings of Armann, Jenkins, and Burton (2015), who observed that prior familiarity with a person negatively affects memory for specific images of that person. We utilized VGG16 neural network models implemented in PyTorch, trained on datasets of facial images to simulate familiarity through prior exposure. The experimental design mirrored the original study's structure, involving Person Recognition and Image Recognition tasks under "Same," "Different," and "Unseen" conditions. We introduced varying levels of noise to model short-term memory fading and tested a range of thresholds for recognition decisions. Our results partially replicated the familiarity effect in the "Different" condition at the fc7 layer, where models performed better with familiar identities in person recognition but achieved higher accuracy with unfamiliar identities in image recognition. However, limitations such as potential overtraining of models, simplistic memory simulations, and the absence of cognitive factors like attention and memory biases affected the completeness of the replication. Differences between the conv5 and fc7 layers highlighted the importance of feature abstraction levels in recognition tasks. The study underscores the complexity of modeling human memory and suggests that integrating more sophisticated cognitive mechanisms could enhance the fidelity of neural network simulations. Future research should focus on refining memory modeling techniques and incorporating cognitive elements to better replicate human memory processes.
